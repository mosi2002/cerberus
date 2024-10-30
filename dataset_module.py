import os
from PIL import Image
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, patches, masks):
        self.patches = patches
        self.masks = masks
        self.normalization = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_patch = self.patches[idx]
        mask_patch = self.masks[idx]
        img_patch = self.normalization(torch.tensor(np.array(img_patch) / 255.0, dtype=torch.float32).permute(2, 0, 1)) 
        mask_patch = torch.tensor(np.array(mask_patch), dtype=torch.long).squeeze() 
        return img_patch, mask_patch
    

class DatasetModule:
    def __init__(self, patch_size, overlap=0, image_dir=None, mask_dir=None):
        self.patch_size = patch_size
        self.overlap = overlap
        self.image_dir = image_dir
        self.mask_dir = mask_dir 
        self.indexes = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        print(self.indexes)

    def _prepare_patching(self, img, mask, input_size, output_size, output_overlap_size):
        """Prepare the patch information for tile processing."""

        def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)

        img_array = np.array(img)
        mask_array = np.array(mask)

        img_shape = img_array.shape[:2]
        mask_shape = mask_array.shape[:2]

        im_h, im_w = img_shape
        m_h, m_w = mask_shape

        # Calculate padding for images
        last_h, _ = get_last_steps(im_h, input_size, output_size)
        last_w, _ = get_last_steps(im_w, input_size, output_size)
        diff = input_size - output_size
        padt = padl = diff // 2
        padb = last_h + input_size - im_h
        padr = last_w + input_size - im_w
        padded_img = np.pad(img_array, ((padt, padb), (padl, padr), (0, 0)), mode="reflect")

        # Calculate padding for masks
        padb_mask = last_h + input_size - m_h
        padr_mask = last_w + input_size - m_w 
        padded_mask = np.pad(mask_array, ((padt, padb_mask), (padl, padr_mask)), mode="reflect")

        input_tl_y = np.arange(0, last_h, output_size, dtype=np.int32)
        input_tl_x = np.arange(0, last_w, output_size, dtype=np.int32)
        input_tl_y, input_tl_x = np.meshgrid(input_tl_y, input_tl_x)
        input_tl = np.stack([input_tl_y.flatten(), input_tl_x.flatten()], axis=-1)
        output_tl = input_tl + diff // 2

        padded_img_shape = padded_img.shape[:2]
        output_br = output_tl + output_size
        input_br = input_tl + input_size
        sel = np.any(input_br > padded_img_shape, axis=-1)
        info_list = np.stack(
            [
                np.stack([input_tl[~sel], input_br[~sel]], axis=1),
                np.stack([output_tl[~sel], output_br[~sel]], axis=1),
            ],
            axis=1,
        )

        if output_overlap_size == 0:
            ovl_output_tl = output_tl + output_overlap_size
            ovl_input_tl = ovl_output_tl - diff // 2
            ovl_output_br = ovl_output_tl + output_size
            ovl_input_br = ovl_input_tl + input_size
            sel = np.any(ovl_input_br > padded_img_shape, axis=-1)
            ovl_info_list = np.stack(
                [
                    np.stack([ovl_input_tl[~sel], ovl_input_br[~sel]], axis=1),
                    np.stack([ovl_output_tl[~sel], ovl_output_br[~sel]], axis=1),
                ],
                axis=1,
            )
            info_list = np.concatenate([info_list, ovl_info_list], axis=0)

        return padded_img, padded_mask, info_list, [padt, padl]

    
    def extract_patches(self, image, mask):
        """
        Extract patches from the image and mask.
        """
        patches = []
        masks = []
        image = image.resize((self.patch_size, self.patch_size))
        mask = mask.resize((self.patch_size, self.patch_size))
        img_array = np.array(image)
        mask_array = np.array(mask)
        padded_img, padded_mask, patch_info, _ = self._prepare_patching(
            img_array,
            mask_array,
            self.patch_size,
            self.patch_size,
            self.overlap
        )

        for patch in patch_info:
            input_tl, input_br = patch[0]
            output_tl, output_br = patch[1]

            img_patch = padded_img[input_tl[0]:input_br[0], input_tl[1]:input_br[1]]
            mask_patch = padded_mask[output_tl[0]:output_br[0], output_tl[1]:output_br[1]]

            patches.append(Image.fromarray(img_patch))
            masks.append(Image.fromarray(mask_patch))

        return patches, masks

    
    def semantic_to_instance(self, segmentation_mask):

        binary_mask = np.zeros(segmentation_mask.shape, dtype=int)
        binary_mask[segmentation_mask == 0] = 1
        return binary_mask
    def glas_semantic_to_instance(self, segmentation_mask):

        binary_mask = np.zeros(segmentation_mask.shape, dtype=int)
        binary_mask[segmentation_mask > 0] = 1
        # plt.figure(figsize=(12, 10))
        # plt.imshow(binary_mask, cmap='gray')
        # plt.show()
        
        return binary_mask

#         num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8))
#         instance_mask = np.zeros_like(binary_mask)
#         for label in range(1, num_labels):  
#             instance_mask[labels == label] = label
        # return instance_mask
    
    def process_file_list(self):
        """
        Process a list of image file paths and extract patches.
        """
        def process_file(index):
            img = Image.open(os.path.join(self.image_dir, index)).convert('RGB')
            mask = cv2.imread(os.path.join(self.mask_dir, index), cv2.IMREAD_GRAYSCALE)
            mask = np.array(mask)  
            # mask = self.semantic_to_instance(mask)
            mask = self.glas_semantic_to_instance(mask)
            mask = mask.astype(np.uint8)
            mask = Image.fromarray(mask)
            # patches, masks = self.extract_patches(img, mask)
            patches = [img.resize((self.patch_size, self.patch_size))]
            masks = [mask.resize((self.patch_size, self.patch_size))]


            return patches, masks

        patch_list = []
        mask_list = []
        if self.image_dir and self.indexes:
            for index in self.indexes:
                patches, masks = process_file(index)
                patch_list.extend(patches)
                mask_list.extend(masks)
        # assert 1 == 2
        return patch_list, mask_list
    
    
    def _post_process_patches(self, patches, predictions):
        pass

