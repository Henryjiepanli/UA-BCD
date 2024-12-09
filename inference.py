import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from PIL import ImageFile
import cv2
from datetime import datetime
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import argparse
from network.UABCD import UABCD, Epistemic_Uncertainty_Estimation
from osgeo import gdal
from osgeo import ogr
from tqdm import tqdm

def png_to_shp(input_png, input_tif, output_shp):
    # 打开.tif文件获取地理信息
    raster_ds = gdal.Open(input_tif)
    if raster_ds is None:
        print("无法打开.tif文件")
        return
    geotransform = raster_ds.GetGeoTransform()  # 获取地理转换信息
    if geotransform is None:
        print("无法获取地理转换信息")
        return
    
    # 获取地理转换参数
    x_origin = geotransform[0]
    y_origin = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    
    # 打开.png文件
    img = cv2.imread(input_png, cv2.IMREAD_GRAYSCALE)
    
    # 创建.shp文件
    driver = ogr.GetDriverByName("ESRI Shapefile")
    vector_ds = driver.CreateDataSource(output_shp)
    if vector_ds is None:
        print("无法创建.shp文件")
        return
    
    # 创建.shp文件的图层
    layer = vector_ds.CreateLayer("layer", None, ogr.wkbPolygon)
    
    # 添加.shp文件的字段
    field_defn = ogr.FieldDefn("value", ogr.OFTInteger)
    layer.CreateField(field_defn)
    threshold = 10
    # 创建.shp文件的要素
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > threshold:  # 根据需要设置阈值，排除面积过小的轮廓
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for point in contour.squeeze():
                if len(point) == 2:  # 确保点包含两个坐标值
                    # 将像素坐标转换为地理坐标
                    x_geo = x_origin + point[0] * pixel_width
                    y_geo = y_origin + point[1] * pixel_height
                    ring.AddPoint(x_geo, y_geo)
            if ring.GetPointCount() > 2:  # 确保轮廓至少包含3个点
                polygon = ogr.Geometry(ogr.wkbPolygon)
                polygon.AddGeometry(ring)
                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetGeometry(polygon)
                layer.CreateFeature(feature)
                feature = None
    # 关闭数据源
    vector_ds = None
    print("转换完成：.png -> .shp")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uabcd_model_path', type=str, default='')
    parser.add_argument('--eue_model_path', type=str, default='')
    parser.add_argument('--A_path', type=str, default='')
    parser.add_argument('--B_path', type=str, default='')
    parser.add_argument('--pos', type=str, default='dongxihuqu')
    parser.add_argument('--batchsize', type=int, default=32)
    opt = parser.parse_args()
    save_path = './Inference_Result/' + opt.pos + '/'
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        net = UABCD(latent_dim=8, num_classes=1).cuda()
        net.load_state_dict(torch.load(opt.uabcd_model_path))

        eue = Epistemic_Uncertainty_Estimation(ndf=64).cuda()
        eue.load_state_dict(torch.load(opt.eue_model_path))

        inference_huge_image(opt=opt, img_A_path=opt.A_path, img_A_path=opt.B_path, model=net, eue=eue,
                             save_path=save_path, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

def inference_huge_image(opt, img_A_path, img_B_path, model, eue, save_path, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    print('Start!')
    model.eval()
    eue.eval()
    img_size = 256
    st = img_size // 2
    batch_size = opt.batchsize  # Batch size for processing patches

    # Load images using PIL and convert to numpy arrays
    full_image_A = np.array(Image.open(img_A_path)).astype(np.float32)
    full_image_B = np.array(Image.open(img_B_path)).astype(np.float32)

    full_height, full_width, _ = full_image_A.shape
    num_h = (full_height - img_size) // st
    num_w = (full_width - img_size) // st

    # Ensure padding values are not negative
    padding_h = max(0, img_size + st * num_h - full_height)
    padding_w = max(0, img_size + st * num_w - full_width)

    # Normalize images
    full_image_A = (full_image_A / 255.0 - mean) / std
    full_image_B = (full_image_B / 255.0 - mean) / std

    # Pad images
    full_image_A = np.pad(full_image_A, ((0, padding_h), (0, padding_w), (0, 0)), mode='constant')
    full_image_B = np.pad(full_image_B, ((0, padding_h), (0, padding_w), (0, 0)), mode='constant')

    # Initialize tensors for predictions
    predict = np.zeros((1, 1, img_size + st * (num_h + 1), img_size + st * (num_w + 1)))
    uncertainty_predict = np.zeros((1, 1, img_size + st * (num_h + 1), img_size + st * (num_w + 1)))
    entropy_predict = np.zeros((1, 1, img_size + st * (num_h + 1), img_size + st * (num_w + 1)))
    count = np.zeros((1, 1, img_size + st * (num_h + 1), img_size + st * (num_w + 1)))

    patches_A = []
    patches_B = []
    coords = []

    for i in range(num_h + 1):
        for j in range(num_w + 1):
            patch_A = full_image_A[i*st: img_size + i*st, j*st: img_size + j*st, :]
            patch_B = full_image_B[i*st: img_size + i*st, j*st: img_size + j*st, :]

            patches_A.append(patch_A)
            patches_B.append(patch_B)
            coords.append((i, j))

            if len(patches_A) == batch_size:
                process_batch(patches_A, patches_B, coords, model, eue, predict, uncertainty_predict, entropy_predict, count, img_size, st)
                patches_A, patches_B, coords = [], [], []

    # Process remaining patches
    if patches_A:
        process_batch(patches_A, patches_B, coords, model, eue, predict, uncertainty_predict, entropy_predict, count, img_size, st)

    predict = predict / count
    uncertainty_predict = uncertainty_predict / count
    entropy_predict = entropy_predict / count

    out_predict = predict[:, :, :full_height, :full_width].squeeze()
    out_predict[out_predict >= 0.5] = 1
    out_predict[out_predict < 0.5] = 0

    out_uncertainty_predict = uncertainty_predict[:, :, :full_height, :full_width].squeeze()
    entropy_predict = entropy_predict[:, :, :full_height, :full_width].squeeze()

    out_uncertainty_predict = torch.sigmoid(torch.from_numpy(out_uncertainty_predict)).numpy()

    final_savepath = save_path + 'predict.png'
    im = Image.fromarray((out_predict * 255).astype(np.uint8))
    im.save(final_savepath)

    un_final_savepath = save_path + 'uncertainty.png'
    estimated_uncertainty = np.uint8(255 * out_uncertainty_predict)
    estimated_uncertainty = cv2.applyColorMap(estimated_uncertainty, cv2.COLORMAP_JET)
    cv2.imwrite(un_final_savepath, estimated_uncertainty)

    entropy_final_savepath = save_path + 'entropy.png'
    entropy = np.uint8(255 * entropy_predict)
    entropy = cv2.applyColorMap(entropy, cv2.COLORMAP_JET)
    cv2.imwrite(entropy_final_savepath, entropy)

    output_shp = save_path + 'changed_building.shp'
    png_to_shp(final_savepath, img_A_path, output_shp)

def process_batch(patches_A, patches_B, coords, model, eue, predict, uncertainty_predict, entropy_predict, count, img_size, st):
    input_A = torch.from_numpy(np.stack(patches_A)).permute(0, 3, 1, 2).float().cuda()
    input_B = torch.from_numpy(np.stack(patches_B)).permute(0, 3, 1, 2).float().cuda()
    # print(input_A.size(),input_B.size())

    # Model inference
    output_patches = model(input_A, input_B)[0]
    estimated_patches = eue(torch.cat((input_A, input_B, torch.sigmoid(output_patches.detach())), 1))
    estimated_patches = F.upsample(estimated_patches, size=(img_size, img_size), mode='bilinear', align_corners=True)
    entropy_patches = -1 * output_patches.sigmoid() * torch.log(output_patches.sigmoid() + 1e-8)
    entropy_patches = (entropy_patches - entropy_patches.min()) / (entropy_patches.max() - entropy_patches.min() + 1e-8)

    for idx, (i, j) in enumerate(tqdm(coords)):
        count[:, :, i * st: img_size + i * st, j * st: img_size + j * st] += 1
        uncertainty_predict[:, :, i * st: img_size + i * st, j * st: img_size + j * st] += estimated_patches[idx].cpu().numpy()
        predict[:, :, i * st: img_size + i * st, j * st: img_size + j * st] += torch.sigmoid(output_patches.detach())[idx].cpu().numpy()
        entropy_predict[:, :, i * st: img_size + i * st, j * st: img_size + j * st] += entropy_patches[idx].cpu().numpy()

    # Clear memory cache
    del input_A, input_B

if __name__ == '__main__':
    time_1 = datetime.now()
    print(time_1)
    main()
    time_2 = datetime.now()
    print(time_2)
    print('cost:', time_2-time_1)
