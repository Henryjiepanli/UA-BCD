import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import numpy as np
from datetime import datetime
from utils import dataloader
from utils.metrics import Evaluator
from utils.tools import adjust_lr, AvgMeter, print_network, poly_lr
import argparse
import logging
from network.UABCD import UABCD, Epistemic_Uncertainty_Estimation
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)

    return annealed

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)

    return l2_reg

# ----------------------------------------------------------------------------------------------------------------------

def Train(train_loader, BCD_Model, EUE_Model, BCD_Model_optimizer, EUE_Model_optimizer, epoch, Eva):
    BCD_Model.train()
    EUE_Model.train()
    loss_record_uabcd = AvgMeter()
    loss_record_eue = AvgMeter()
    print('UABCD Learning Rate: {}'.format(BCD_Model_optimizer.param_groups[0]['lr']))
    print('EUE Learning Rate: {}'.format(EUE_Model_optimizer.param_groups[0]['lr']))
    for i, sample in enumerate(tqdm(train_loader), start=1):
        BCD_Model_optimizer.zero_grad()
        EUE_Model_optimizer.zero_grad()
        A, B, mask = sample['A'], sample['B'], sample['label']
        A = Variable(A)
        B = Variable(B)
        gts = Variable(mask)
        A = A.cuda()
        B = B.cuda()
        Y = gts.cuda()
        gts = Y.unsqueeze(1)

        # train UABCD
        p_m_prior, p_m_post, p_a, latent_loss = BCD_Model(A, B, gts)
        reg_loss = l2_regularisation(BCD_Model.AUEM_prior) + l2_regularisation(BCD_Model.AUEM_post) + \
                   l2_regularisation(BCD_Model.decoder_prior) + l2_regularisation(BCD_Model.decoder_post)
        reg_loss = opt.reg_weight * reg_loss
        anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)
        latent_loss = opt.lat_weight * anneal_reg * latent_loss
        loss_cd = BCE_loss(p_m_post, gts) + BCE_loss(p_m_prior, gts) + BCE_loss(p_a, gts)
        seg_loss = loss_cd + reg_loss + opt.vae_loss_weight * latent_loss 
        seg_loss.backward()
        BCD_Model_optimizer.step()

        # get variance map (entropy)
        preds = [torch.sigmoid(p_m_post)]
        with torch.no_grad():
            for ff in range(opt.forward_iter - 1):
                ff_m = BCD_Model(A,B, gts)[1]
                preds.append(torch.sigmoid(ff_m))
        preds = torch.cat(preds, dim=1)
        mean_preds = torch.mean(preds, 1, keepdim=True)
        var_map = -1 * mean_preds * torch.log(mean_preds + 1e-8)
        var_map = (var_map - var_map.min()) / (var_map.max() - var_map.min() + 1e-8)
        var_map = Variable(var_map.data, requires_grad=True)

        # train EUE
        output_D = EUE_Model(torch.cat((A, B, torch.sigmoid(p_m_post.detach())), 1))
        output_D = F.upsample(output_D, size=(opt.trainsize, opt.trainsize), mode='bilinear', align_corners=True)
        approximation_loss = BCE_loss(output_D, var_map)
        approximation_loss.backward()
        EUE_Model_optimizer.step()


        loss_record_uabcd.update(seg_loss.data, opt.batchsize)
        loss_record_eue.update(approximation_loss.data, opt.batchsize)

        if i % 100 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Seg Loss: {:.4f}, Consist Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record_uabcd.show(), loss_record_eue.show()))
            logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Seg Loss: {:.4f}, Consist Loss: {:.4f}'.
                         format(epoch, opt.epoch, i, total_step, loss_record_uabcd.show(), loss_record_eue.show()))
            
        output = p_m_post.sigmoid().data.cpu().numpy().squeeze()
        output[output>=0.5] = 1
        output[output<0.5] = 0
        target = Y.cpu().numpy()
        # Add batch sample into evaluator
        # print(target.shape, output.shape)
        Eva.add_batch(target, output.astype(np.int64))
    IoU = Eva.Intersection_over_Union()[1]
    F1 = Eva.F1()[1]
    print('Epoch [{:03d}/{:03d}], \n[Training] IoU: {:.4f}, F1: {:.4f}'.format(epoch, opt.epoch, IoU, F1))

    logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], IoU: {:.4f}, F1: {:.4f}'.format(epoch, opt.epoch, IoU, F1))

def Val(test_loader, BCD_Model, EUE_Model, epoch, Eva, save_path):
    global best_f1, best_epoch
    BCD_Model.eval()
    EUE_Model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader)):
            A, B, mask = sample['A'], sample['B'], sample['label']
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()
            res = BCD_Model(A, B)[0]
            output = F.upsample(res, size=(opt.trainsize, opt.trainsize), mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output[output>=0.5] = 1
            output[output<0.5] = 0
            target = Y.cpu().numpy()
            # Add batch sample into evaluator
            Eva.add_batch(target, output.astype(np.int64))
    IoU = Eva.Intersection_over_Union()[1]
    F1 = Eva.F1()[1]

    print('Epoch [{:03d}/{:03d}], \n[Validing] IoU: {:.4f}, F1: {:.4f}'.format(epoch, opt.epoch, IoU, F1))
    logging.info('#Val#:Epoch:{} IoU:{} F1:{}'.format(epoch, IoU, F1))
    new_f1 = F1
    if new_f1 >= best_f1:
        best_f1 = new_f1
        best_epoch = epoch
        print('Best Model Iou :%.4f; F1 : %.4f; Best epoch : %d' % (IoU, F1, best_epoch))

        torch.save(BCD_Model.state_dict(), save_path + 'Seg_epoch_best.pth')
        torch.save(EUE_Model.state_dict(), save_path + 'EUE_epoch_best.pth')

    logging.info('#TEST#:Epoch:{} F1:{} bestEpoch:{} bestF1:{}'.format(epoch, F1, best_epoch, best_f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr_uabcd', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_eue', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--latent_dim', type=int, default=8, help='latent dimension')
    parser.add_argument('--forward_iter', type=int, default=5, help='number of iterations of UABCD forward')
    parser.add_argument('--lat_weight', type=float, default=1.0, help='weight for latent loss')
    parser.add_argument('--vae_loss_weight', type=float, default=2, help='weight for vae loss')
    parser.add_argument('--reg_weight', type=float, default=1e-4, help='weight for regularization term')
    parser.add_argument('--data_name', type=str, default='LEVIR',
                        help='the test rgb images root')
    parser.add_argument('--segclass', type=int, default=1,
                        help='')
    parser.add_argument('--save_path', type=str,
                            default='./train_output/UABCD/')
    opt = parser.parse_args()

    save_path = opt.save_path + opt.data_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('UABCD Learning Rate: {}'.format(opt.lr_uabcd))
    print('PUA Learning Rate: {}'.format(opt.lr_eue))

    # build models
    BCD_Model = UABCD(latent_dim=opt.latent_dim, num_classes=opt.segclass)
    BCD_Model.cuda()
    BCD_Model_params = BCD_Model.parameters()
    BCD_Model_optimizer = torch.optim.Adam(BCD_Model_params, opt.lr_uabcd)

    EUE_Model = Epistemic_Uncertainty_Estimation(ndf=64)
    EUE_Model.cuda()
    EUE_Model_params = EUE_Model.parameters()
    EUE_Model_optimizer = torch.optim.Adam(EUE_Model_params, opt.lr_eue)

    # set path
    if opt.data_name == 'LEVIR':
        opt.train_root = './Data/Change_Detection/LEVIR-CD_cropped256/train/' 
        opt.val_root = './Data/Change_Detection/LEVIR-CD_cropped256/val/'
        palatte = [[0,0,0], [255,255,255]]
    elif opt.data_name == 'Google':
        opt.train_root = './Data/Change_Detection/Google-CD/train/' 
        opt.val_root = './Data/Change_Detection/Google-CD/val/'
        palatte = [[0,0,0], [255,255,255]]
    elif opt.data_name == 'WHU':
        opt.train_root = './Data/Change_Detection/WHU-CD256-HANet/WHU-CD256-HANet/train/' 
        opt.val_root = './Data/Change_Detection/WHU-CD256-HANet/WHU-CD256-HANet/val/'
        palatte = [[0,0,0], [255,255,255]]
    elif opt.data_name == 'SYSU':
        opt.train_root = './Data/Change_Detection/SYSU-CD/train/' 
        opt.val_root = './Data/Change_Detection/SYSU-CD/val/'
        palatte = [[0,0,0], [255,255,255]]
    elif opt.data_name == 'Lebedev':
        opt.train_root = './Data/Change_Detection/Lebedev/train/' 
        opt.val_root = './Data/Change_Detection/Lebedev/val/'
        palatte = [[0,0,0], [255,255,255]]


    train_loader = dataloader.get_loader(img_A_root = opt.train_root + 'A/', img_B_root = opt.train_root + 'B/', gt_root = opt.train_root + 'label/', trainsize = opt.trainsize, palatte = palatte, mode ='train', batchsize = opt.batchsize, mosaic_ratio=0.25, num_workers=4, shuffle=True, pin_memory=True)
    test_loader = dataloader.get_loader(img_A_root = opt.val_root + 'A/', img_B_root = opt.val_root + 'B/', gt_root = opt.val_root + 'label/', trainsize = opt.trainsize, palatte = palatte, mode ='val', batchsize = opt.batchsize, mosaic_ratio=0, num_workers=4, shuffle=False, pin_memory=True)
    total_step = len(train_loader)

    logging.basicConfig(filename=save_path+'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("UABCD-Train")
    logging.info("Config")
    logging.info('epoch:{}; lr_uabcd:{}; lr_eue:{}; batchsize:{}; trainsize:{}; save_path:{}\
                lat_weight:{} vae_loss_weight: {} reg_loss_weight:{}'.
                format(opt.epoch, opt.lr_uabcd, opt.lr_eue, opt.batchsize, opt.trainsize, save_path, opt.lat_weight,\
                        opt.vae_loss_weight, opt.reg_weight))

    # loss function
    BCE_loss = torch.nn.BCEWithLogitsLoss().cuda()
    print("Let's go!")
    best_f1 = 0
    best_epoch = 0
    Eva_tr = Evaluator(2)
    Eva_val = Evaluator(2)
    for epoch in range(1, (opt.epoch+1)):
        Eva_tr.reset()
        Eva_val.reset()
        uabcd_lr = adjust_lr(BCD_Model_optimizer, opt.lr_uabcd, epoch, 0.1, opt.decay_epoch)
        eue_lr = adjust_lr(EUE_Model_optimizer, opt.lr_eue, 0.1, opt.decay_epoch)
        Train(train_loader, BCD_Model, EUE_Model, BCD_Model_optimizer, EUE_Model_optimizer, epoch, Eva_tr)
        Val(test_loader, BCD_Model, EUE_Model, epoch, Eva_val, save_path)



