import os
import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import scipy.ndimage as ndi

ds2task_ids = {
    'BUSI' : 0,
    'STU' : 1,
    'TestSet' : 2,
    'KUtest' : 2, 
    'UDIAT' : 3,
    'QAMEBI' : 4
}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, ids, root, num_classes, mode, transform=None):
        self.imgs    = ids['images']
        self.labels  = ids['labels']
        self.root = root,
        self.num_classes = num_classes
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.imgs)

    def get_boundary_map(self, gt):
        #gt = cv2.resize(gt, (352, 352))
        #gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        gt = gt[:,:,0]                                  # [256,256,1] -> [256,256]
        dis = ndi.distance_transform_edt(gt)
        dis_in = ndi.distance_transform_edt(1-gt)
        dis[gt == 0] = dis_in[gt == 0]
        dis = np.exp(-1.0 * (dis-1))
        return dis[None]
    
    def __getitem__(self, idx):
        img_name, mask_name = self.imgs[idx], self.labels[idx]
        
        img = cv2.imread(f'./inputs/{img_name}')
        img = cv2.resize(img, (256, 256))
        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(f'./inputs/{mask_name}', cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)
        mask = np.array(Image.fromarray(mask[:,:,0]).resize((256, 256), resample=Image.NEAREST))[:,:,None]
        #mask = cv2.resize(mask, (256,256))
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
        distancemap = self.get_boundary_map(mask)
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        task_id = ds2task_ids[img_name.split('/')[0]]
        if img.shape[1:] != mask.shape[1:]:
            print(img)
        
        dics = {'image' : img, 'mask': mask}
        if self.mode == 'with_boundary':    
            dics['boundary_distance_map'] = distancemap      # [256, 256]
        
        return dics, {'img_id': img_name.split('/')[-1].replace('.png', ''), 'task_id' : task_id}
       # return img, mask, {'img_id': img_name.split('/')[-1].replace('.png', ''), 'task_id' : task_id}

class Dataset2(torch.utils.data.Dataset):
    def __init__(self, ids, root, img_ext, mask_ext, num_classes, transform=None):
        self.imgs    = ids['images']
        self.root = root,
        # self.img_ids = img_ids
        # self.img_dir = img_dir
        # self.mask_dir = mask_dir
        # self.img_ext = img_ext
        # self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        
        img = cv2.imread(f'./inputs/{img_name}')
        img = cv2.resize(img, (256, 256))
       
        if self.transform is not None:
            augmented = self.transform(image=img, mask=img)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        
        #task_id = ds2task_ids[img_name.split('/')[0]]
       
        return img, {'img_id': img_name.split('/')[-1].replace('.png', ''), 'task_id' : 0}


class Dataset_with_dis(torch.utils.data.Dataset):
    def __init__(self, ids, root, img_ext, mask_ext, num_classes, transform=None):
        self.imgs    = ids['images']
        self.labels  = ids['labels']
        self.root = root,
        # self.img_ids = img_ids
        # self.img_dir = img_dir
        # self.mask_dir = mask_dir
        # self.img_ext = img_ext
        # self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def get_boundary_map(self, gt):
        #gt = cv2.resize(gt, (352, 352))
        #gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        gt = gt[:,:,0]                                  # [256,256,1] -> [256,256]
        dis = ndi.distance_transform_edt(gt)
        dis_in = ndi.distance_transform_edt(1-gt)
        dis[gt == 0] = dis_in[gt == 0]
        dis = np.exp(-1.0 * (dis-1))
        return dis[None]
     
    def __getitem__(self, idx):
        img_name, mask_name = self.imgs[idx], self.labels[idx]
        
        img = cv2.imread(f'./inputs/{img_name}')
        img = cv2.resize(img, (256, 256))
        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(f'./inputs/{mask_name}', cv2.IMREAD_GRAYSCALE)[..., None])
            # mask.append(cv2.imread(os.path.join(self.mask_dir, #str(i),
            #             img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)
        mask = np.array(Image.fromarray(mask[:,:,0]).resize((256, 256), resample=Image.NEAREST))[:,:,None]
        #mask = cv2.resize(mask, (256,256))
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        #! get distance map
        distance_map = self.get_boundary_map(mask)      # [256, 256]
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        task_id = ds2task_ids[img_name.split('/')[0]]
        if img.shape[1:] != mask.shape[1:]:
            print(img)
        return img, mask, {'img_id': img_name.split('/')[-1].replace('.png', ''), 'task_id' : task_id, 'distance_map' : distance_map}
