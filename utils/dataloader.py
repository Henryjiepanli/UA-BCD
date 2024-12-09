import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
from utils import custom_transforms as tr

def randomCrop_Mosaic(image_A, image_B, label, crop_win_width, crop_win_height):
    image_width = image_A.size[0]
    image_height = image_B.size[1]
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image_A.crop(random_region), image_B.crop(random_region), label.crop(random_region)


def mask_to_onehot(mask, palette):

    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1) #单通道有索引使用
        # class_map = equality.astype(int)  #单通道无索引使用
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    semantic_map = np.argmax(semantic_map,axis=-1)
    return semantic_map


class Multi_Class_Segmentation_Dataset(data.Dataset):
    def __init__(self, img_A_root, img_B_root, gt_root, trainsize, palatte, mode, mosaic_ratio=0.25):
        self.trainsize = trainsize
        self.image_A_root = img_A_root
        self.image_B_root = img_B_root
        self.palatte = palatte
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.gt_root = gt_root
        self.images_A = [self.image_A_root + f for f in os.listdir(self.image_A_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.images_B = [self.image_B_root + f for f in os.listdir(self.image_B_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.gts = [self.gt_root + f for f in os.listdir(self.gt_root) if f.endswith('.jpg') \
                    or f.endswith('.png')]
        self.images_A = sorted(self.images_A)
        self.images_B = sorted(self.images_B)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images_A)

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio:
            image_A, image_B, mask = self.load_img_and_mask(index)
        else:
            image_A, image_B, mask = self.load_mosaic_img_and_mask(index)

        sample = {'A': image_A, 'B': image_B, 'label': mask}

        if self.mode == 'train':
            return self.transform_tr(sample)
        elif self.mode == 'val':
            return self.transform_val(sample)
        elif self.mode == 'test':
            file_name = self.images_A[index].split('/')[-1][:-len(".tif")]
            return self.transform_test(sample),file_name
        
    def load_img_and_mask(self, index):
        image_A = Image.open(self.images_A[index]).convert('RGB')
        image_B = Image.open(self.images_B[index]).convert('RGB')
        mask = np.array(Image.open(self.gts[index]).convert('RGB'), dtype=np.uint8)
        mask = mask_to_onehot(mask, self.palatte)
        mask = Image.fromarray(np.uint8(mask))
        return image_A, image_B, mask

    def load_mosaic_img_and_mask(self, index):
       indexes = [index] + [random.randint(0, self.size - 1) for _ in range(3)]
       img_a_A, img_a_B,  mask_a = self.load_img_and_mask(indexes[0])
       img_b_A, img_b_B, mask_b = self.load_img_and_mask(indexes[1])
       img_c_A, img_c_B, mask_c = self.load_img_and_mask(indexes[2])
       img_d_A, img_d_B, mask_d = self.load_img_and_mask(indexes[3])

       w = self.trainsize
       h = self.trainsize

       start_x = w // 4
       strat_y = h // 4
        # The coordinates of the splice center
       offset_x = random.randint(start_x, (w - start_x))
       offset_y = random.randint(strat_y, (h - strat_y))


       crop_size_a = (offset_x, offset_y)
       crop_size_b = (w - offset_x, offset_y)
       crop_size_c = (offset_x, h - offset_y)
       crop_size_d = (w - offset_x, h - offset_y)

       croped_a_A, croped_a_B, mask_crop_a = randomCrop_Mosaic(img_a_A.copy(), img_a_B.copy(), mask_a.copy(),crop_size_a[0], crop_size_a[1]) 
       croped_b_A, croped_b_B, mask_crop_b = randomCrop_Mosaic(img_b_A.copy(), img_b_B.copy(), mask_b.copy(),crop_size_b[0], crop_size_b[1])
       croped_c_A, croped_c_B, mask_crop_c = randomCrop_Mosaic(img_c_A.copy(), img_c_B.copy(), mask_c.copy(),crop_size_c[0], crop_size_c[1])
       croped_d_A, croped_d_B, mask_crop_d = randomCrop_Mosaic(img_d_A.copy(), img_d_B.copy(), mask_d.copy(),crop_size_d[0], crop_size_d[1])

       croped_a_A, croped_a_B, mask_crop_a = np.array(croped_a_A), np.array(croped_a_B), np.array(mask_crop_a)
       croped_b_A, croped_b_B, mask_crop_b = np.array(croped_b_A), np.array(croped_b_B), np.array(mask_crop_b)
       croped_c_A, croped_c_B, mask_crop_c = np.array(croped_c_A), np.array(croped_c_B), np.array(mask_crop_c)
       croped_d_A, croped_d_B, mask_crop_d = np.array(croped_d_A), np.array(croped_d_B), np.array(mask_crop_d)

       top_A = np.concatenate((croped_a_A, croped_b_A), axis=1)
       bottom_A = np.concatenate((croped_c_A, croped_d_A), axis=1)
       img_A = np.concatenate((top_A, bottom_A), axis=0)

       top_B = np.concatenate((croped_a_B, croped_b_B), axis=1)
       bottom_B = np.concatenate((croped_c_B, croped_d_B), axis=1)
       img_B = np.concatenate((top_B, bottom_B), axis=0)


       top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
       bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
       mask = np.concatenate((top_mask, bottom_mask), axis=0)
       mask = np.ascontiguousarray(mask)

       img_A = np.ascontiguousarray(img_A)
       img_B = np.ascontiguousarray(img_B)

       img_A = Image.fromarray(img_A)
       img_B = Image.fromarray(img_B)
       mask = Image.fromarray(mask)

       return img_A, img_B, mask
    
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomGaussianBlur(),
            tr.FixScaleCrop(crop_size=self.trainsize),
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.trainsize),
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_test(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.trainsize),
            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            tr.ToTensor()])

        return composed_transforms(sample)


    def filter_files(self):
        assert len(self.images_A) == len(self.gts)
        assert len(self.images_A) == len(self.images_B)
        images_A = []
        images_B = []
        gts = []
        for img_A_path, img_B_path, gt_path in zip(self.images_A, self.images_B, self.gts):
            img_A = Image.open(img_A_path)
            img_B = Image.open(img_B_path)
            gt = Image.open(gt_path)
            if img_A.size == img_B.size:
                if img_A.size == gt.size:
                    images_A.append(img_A_path)
                    images_B.append(img_B_path)
                    gts.append(gt_path)

        self.images_A = images_A
        self.images_B = images_B
        self.gts = gts


    def __len__(self):
        return self.size

def get_loader(img_A_root, img_B_root, gt_root, trainsize, palatte, mode, batchsize, mosaic_ratio=0.25, num_workers=4, shuffle=True, pin_memory=True):

    dataset = Multi_Class_Segmentation_Dataset(img_A_root = img_A_root, img_B_root = img_B_root, gt_root = gt_root, trainsize = trainsize, palatte = palatte, mode = mode, mosaic_ratio=mosaic_ratio)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader



# if __name__ == '__main__':
#     root = '/home/lijiepan/multi_class_semantic_segementation/drive_dataset/train/'
#     batchsize = 8
#     trainsize = 512
#     palette = [[255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 255], [0, 0, 0]]
#     data = get_loader(root, batchsize, trainsize, palette, num_workers=4, shuffle=True, pin_memory=True)
#     for x,y in data:
#         print(x)
#         print(y)



