import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .backbones.pvtv2 import *
from torch.distributions import Normal, Independent, kl
import warnings
warnings.filterwarnings('ignore')


class Aleatoric_Uncertainty_Estimation_Module_Prior(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Aleatoric_Uncertainty_Estimation_Module_Prior, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1 = nn.Linear(channels * 8 * 8 * 8, latent_size)  # adjust according to input size
        self.fc2 = nn.Linear(channels * 8 * 8 * 8, latent_size)  # adjust according to input size

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input_feature):
        output = self.leakyrelu(self.bn1(self.layer1(input_feature)))
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        output = self.leakyrelu(self.bn5(self.layer5(output)))
        output = output.view(-1, self.channel * 8 * 8 * 8)  # adjust according to input size

        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)

        return mu, logvar, dist


class Aleatoric_Uncertainty_Estimation_Module_Post(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Aleatoric_Uncertainty_Estimation_Module_Post, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1 = nn.Linear(channels * 8 * 8 * 8, latent_size)  # adjust according to input size
        self.fc2 = nn.Linear(channels * 8 * 8 * 8, latent_size)  # adjust according to input size

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input_feature):
        output = self.leakyrelu(self.bn1(self.layer1(input_feature)))
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        output = self.leakyrelu(self.bn5(self.layer5(output)))
        output = output.view(-1, self.channel * 8 * 8 * 8)  # adjust according to input size

        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)

        return mu, logvar, dist


class Epistemic_Uncertainty_Estimation(nn.Module):
    def __init__(self, ndf):
        super(Epistemic_Uncertainty_Estimation, self).__init__()
        self.conv1 = nn.Conv2d(7, ndf, kernel_size=3, stride=2, padding=1)  # 4 for predictive uncertainty
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(ndf)
        self.bn2 = nn.BatchNorm2d(ndf)
        self.bn3 = nn.BatchNorm2d(ndf)
        self.bn4 = nn.BatchNorm2d(ndf)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x


class UABCD(nn.Module):
    def __init__(self,latent_dim, num_classes):
        super(UABCD, self).__init__()
        channel = 128

        self.backbone = pvt_v2_b2()  
        path = './pretrained_model/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.conv1 = BasicConv2d(2*64, 64, 1)
        self.conv2 = BasicConv2d(2*128, 128, 1)
        self.conv3 = BasicConv2d(2*320, 320, 1)
        self.conv4 = BasicConv2d(2*512, 512, 1)

        self.conv_4 = BasicConv2d(512, channel, 3, 1, 1)
        self.conv_3 = BasicConv2d(320, channel, 3, 1, 1)
        self.conv_2 = BasicConv2d(128, channel, 3, 1, 1)
        self.conv_1 = BasicConv2d(64, channel, 3, 1, 1)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.coarse_out = nn.Sequential(nn.Conv2d(4 * channel, channel, kernel_size=3, stride=1, padding=1, bias=True),\
                                      nn.Conv2d(channel, 1, kernel_size=1, stride=1, bias=True))
        self.sigmoid = nn.Sigmoid()

        self.AUEM_prior = Aleatoric_Uncertainty_Estimation_Module_Prior(6, int(channel / 8), latent_dim)
        self.AUEM_post = Aleatoric_Uncertainty_Estimation_Module_Post(7, int(channel / 8), latent_dim)

        self.decoder_prior = Refined_Change_Map_Generation(latent_dim, num_classes)
        self.decoder_post = Refined_Change_Map_Generation(latent_dim, num_classes)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu)

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)

        return kl_div
    
    def Coarse_Change_Map_Generation(self, A, B):
        layer_1_A, layer_2_A, layer_3_A, layer_4_A = self.backbone(A)
        layer_1_B, layer_2_B, layer_3_B, layer_4_B = self.backbone(B)

        layer_1 = self.conv_1(self.conv1(torch.cat((layer_1_A, layer_1_B), dim=1)))
        layer_2 = self.conv_2(self.conv2(torch.cat((layer_2_A, layer_2_B), dim=1)))
        layer_3 = self.conv_3(self.conv3(torch.cat((layer_3_A, layer_3_B), dim=1)))
        layer_4 = self.conv_4(self.conv4(torch.cat((layer_4_A, layer_4_B), dim=1)))

        Fusion = torch.cat((self.upsample8(layer_4), self.upsample4(layer_3), self.upsample2(layer_2), layer_1), 1)
        Guidance_out = self.coarse_out(Fusion)
        Coarse_out = self.upsample4(Guidance_out)

        return Guidance_out, Coarse_out, layer_1, layer_2, layer_3, layer_4


    def forward(self, A, B, y=None):
        Guidance_out, Coarse_out, layer1, layer2, layer3, layer4 = self.Coarse_Change_Map_Generation(A,B)

        Changed_Guidance = self.sigmoid(Guidance_out)
        Non_Changed_Guidance = 1 - self.sigmoid(Guidance_out)

        if y == None:
            mu_prior, logvar_prior, _ = self.AUEM_prior(torch.cat((A, B),1))
            z_prior = self.reparametrize(mu_prior, logvar_prior)
            Refined_out_prior = self.decoder_prior(Changed_Guidance, Non_Changed_Guidance, layer4, layer3, layer2, layer1, z_prior)

            return Refined_out_prior, Coarse_out
        else:
            mu_prior, logvar_prior, dist_prior = self.AUEM_prior(torch.cat((A, B),1))
            z_prior = self.reparametrize(mu_prior, logvar_prior)
            mu_post, logvar_post, dist_post = self.AUEM_post(torch.cat((A, B, y),1))
            z_post = self.reparametrize(mu_post, logvar_post)
            kld = torch.mean(self.kl_divergence(dist_post, dist_prior))

            Refined_out_prior = self.decoder_prior(Changed_Guidance, Non_Changed_Guidance, layer4, layer3, layer2, layer1, z_prior)
            Refined_out_post = self.decoder_post(Changed_Guidance, Non_Changed_Guidance, layer4, layer3, layer2, layer1, z_post)

            return Refined_out_prior, Refined_out_post, Coarse_out, kld


class Refined_Change_Map_Generation(nn.Module):
    def __init__(self,
                 latent_dim, num_classes
                 ):
        super(Refined_Change_Map_Generation, self).__init__()
        channel = 128

        self.down8 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True)
        self.down4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.down2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.noise_conv = nn.Conv2d(channel + latent_dim, channel, kernel_size=1, padding=0)
        self.spatial_axes = [2, 3]

        self.KGFEM4 = Knowledge_Guided_Feature_Enhancement_Module()
        self.KGFEM3 = Knowledge_Guided_Feature_Enhancement_Module()
        self.KGFEM2 = Knowledge_Guided_Feature_Enhancement_Module()
        self.KGFEM1 = Knowledge_Guided_Feature_Enhancement_Module()

        self.Fusion4 = AggUnit(channel)
        self.Fusion3 = AggUnit(channel)
        self.Fusion2 = AggUnit(channel)
        self.Fusion1 = AggUnit(channel)

        self.out_conv = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0),
        )

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            device)
        return torch.index_select(a, dim, order_index)

    def forward(self, Changed_Guidance, Non_Changed_Guidance, layer4, layer3, layer2, layer1, z):
        z_noise = torch.unsqueeze(z, 2)
        z_noise = self.tile(z_noise, 2, layer4.shape[self.spatial_axes[0]])
        z_noise = torch.unsqueeze(z_noise, 3)
        z_noise = self.tile(z_noise, 3, layer4.shape[self.spatial_axes[1]])

        layer4 = torch.cat((layer4, z_noise), 1)
        layer4 = self.noise_conv(layer4)

        layer4 = self.KGFEM4(layer4, self.down8(Changed_Guidance), self.down8(Non_Changed_Guidance))
        layer3 = self.KGFEM3(layer3, self.down4(Changed_Guidance), self.down4(Non_Changed_Guidance))
        layer2 = self.KGFEM2(layer2, self.down2(Changed_Guidance), self.down2(Non_Changed_Guidance))
        layer1 = self.KGFEM1(layer1, Changed_Guidance, Non_Changed_Guidance)

        Fusion = self.Fusion4(layer4)
        Fusion = self.Fusion3(Fusion, layer3)
        Fusion = self.Fusion2(Fusion, layer2)
        Fusion = self.Fusion1(Fusion, layer1)
        Refined_out = self.out_conv(Fusion)

        return Refined_out


class Knowledge_Guided_Feature_Enhancement_Module(nn.Module):
    def __init__(self,):
        super(Knowledge_Guided_Feature_Enhancement_Module, self).__init__()

        self.conv_p = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.conv_n = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, w_p, w_n):
        x_p = x * w_p

        x_n = x * w_n
        max_out_p, _ = torch.max(x_p, dim=1, keepdim=True)
        avg_out_p = torch.mean(x_p, dim=1, keepdim=True)
        spatial_out_p = self.sigmoid(self.conv_p(torch.cat([max_out_p, avg_out_p], dim=1)))
        x_p = spatial_out_p * x_p

        max_out_n, _ = torch.max(x_n, dim=1, keepdim=True)
        avg_out_n = torch.mean(x_n, dim=1, keepdim=True)
        spatial_out_n = self.sigmoid(self.conv_n(torch.cat([max_out_n, avg_out_n], dim=1)))
        x_n = spatial_out_n * x_n
        return x + x_p + x_n


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super(ResidualConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class AggUnit(nn.Module):
    def __init__(self, features):
        super(AggUnit, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = F.interpolate( output, scale_factor=2, mode="bilinear", align_corners=True)

        return output


if __name__ == '__main__':
    A = torch.rand(4,3,256,256).cuda()
    B = torch.rand(4,3,256,256).cuda()

    model = UABCD(latent_dim=8, num_classes=1).cuda()

    outs = model(A,B)

    for o in outs:
        print(o.shape)

