
import os
import yaml
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# from torchsummary import summary
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import torchvision.transforms as transforms
from dataset_module import DatasetModule, CustomDataset
from models.net_desc import create_model
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from models.utils.loss_utils import xentropy_loss, focal_loss, dice_loss, Dice_loss_v2, TverskyLoss, FocalTverskyLoss, btc_loss
from models.backbone import UnetEncoder
from sklearn.model_selection import KFold
from metrics import ConfusionMatrix, dice, jaccard, precision, recall, accuracy, fscore

class run_train_cer():
    def __init__(self):
        # Load configuration
        self.count_res = 0
        with open('train_loop/settings.yml') as f:
            self.config = yaml.full_load(f)
        
    def load_data(self):
        # Dataset and DataLoader setup
        self.dataset_module = DatasetModule(
            patch_size=self.config['dataset_kwargs']['input_shape'],
            overlap=0,
            image_dir=self.config['dataset_kwargs']['input_dir'],
            mask_dir=self.config['dataset_kwargs']['mask_dir']
        )

        # Process files to get patches and masks
        patches, masks = self.dataset_module.process_file_list()

        # Augment data
        augmented_data = self.augment_data(patches, masks)
        patches, masks = augmented_data

        # Initialize K-Fold
        kf = KFold(n_splits=self.config.get('num_folds', 5), shuffle=True, random_state=42)

        # Store loaders for each fold
        self.fold_loaders = []

        for fold_index, (train_indices, val_indices) in enumerate(kf.split(patches)):
            # Get training and validation data for the current fold
            train_patches = patches[train_indices]
            train_masks = masks[train_indices]
            val_patches = patches[val_indices]
            val_masks = masks[val_indices]

            # Create datasets
            train_dataset = CustomDataset(train_patches, train_masks)
            val_dataset = CustomDataset(val_patches, val_masks)

            # Create DataLoaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['loader_kwargs']['train']['batch_size'],
                num_workers=self.config['loader_kwargs']['train']['nr_procs'],
                shuffle=True
            )

            valid_loader = DataLoader(
                val_dataset,
                batch_size=self.config['loader_kwargs']['valid']['batch_size'],
                num_workers=self.config['loader_kwargs']['valid']['nr_procs'],
                shuffle=False
            )

            # Store the loaders for the current fold
            self.fold_loaders.append((train_loader, valid_loader))

            print(f"Fold {fold_index + 1}:")
            print(f"Size of the training dataset: {len(train_dataset)}    Size of the validation dataset: {len(val_dataset)}")
            print(f"Number of batches in the training loader: {len(train_loader)}     Number of batches in the validation loader: {len(valid_loader)}")
    
    def load_model(self):
        # Loading the model
        self.model_args = self.config['model_kwargs']
        self.model = create_model(**self.model_args)
    
    def loding_model_weights(self):
        self.weights = torch.load(self.config['weights_path'])
        self.checkpoint = self.weights['desc']
        
        # Remove 'module.' prefix from the state dictionary keys
        self.checkpoint = {k.replace('module.', ''): v for k, v in self.checkpoint.items()}
        # self.model.load_state_dict(self.checkpoint, strict=False)
        
        backbone_weights = {k: v for k, v in self.checkpoint.items() if k.startswith('backbone')}
        conv_map_weights = {k: v for k, v in self.checkpoint.items() if k.startswith('conv_map')}
    
        selected_weights = {**backbone_weights, **conv_map_weights}
        self.model.load_state_dict(selected_weights, strict=False)
        
       
         # Freeze encoder (backbone)
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # Freeze all decoders except for 'Gland'
        for decoder_name, decoder in self.model.decoder_head.items():
            if decoder_name != 'Gland':
                for param in decoder.parameters():
                    param.requires_grad = False

        # Freeze all output heads except for 'Gland'
        for decoder_name, output_head in self.model.output_head.items():
            if decoder_name != 'Gland':
                for param in output_head.parameters():
                    param.requires_grad = False
        
    def save_model(self, checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        torch.save(self.model.state_dict(), checkpoint_path)
 
    def augment_data(self, images, masks, augment_ratio=0.3, single_aug=True):
        augmented_images = []
        augmented_masks = []

        # Keep all original images and masks
        augmented_images.extend(images)
        augmented_masks.extend(masks)

        # Calculate number of images to augment
        num_images = len(images)
        num_to_augment = int(num_images * augment_ratio)

        # Randomly select images to augment
        indices_to_augment = np.random.choice(num_images, size=num_to_augment, replace=False)

        for idx in indices_to_augment:
            img_pil = images[idx]
            mask_pil = masks[idx]

            # Store original before augmentation
            temp_images = [np.array(img_pil)]
            temp_masks = [np.array(mask_pil)]

            # Randomly choose whether to apply one or multiple augmentations
            if single_aug:  # Apply one augmentation
                aug_choice = np.random.choice(['flip', 'rotate', 'gaussian_blur', 'median_blur', 'color_perturbation'])
                if aug_choice == 'flip':
                    img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
                elif aug_choice == 'rotate':
                    angle = np.random.randint(-30, 30)
                    img_pil = img_pil.rotate(angle, fillcolor=(0, 0, 0))
                    mask_pil = mask_pil.rotate(angle, fillcolor=0)
                elif aug_choice == 'gaussian_blur':
                    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=np.random.rand() * 2))
                elif aug_choice == 'median_blur':
                    img_np = np.array(img_pil)
                    img_np = cv2.medianBlur(img_np, ksize=5)
                    img_pil = Image.fromarray(img_np)
                elif aug_choice == 'color_perturbation':
                    enhancer = ImageEnhance.Color(img_pil)
                    img_pil = enhancer.enhance(np.random.uniform(0.5, 1.5))

            else:  # Apply multiple augmentations
                if np.random.rand() > 0.5:  # Random flip
                    img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)

                if np.random.rand() > 0.5:  # Random rotation
                    angle = np.random.randint(-30, 30)
                    img_pil = img_pil.rotate(angle, fillcolor=(0, 0, 0))
                    mask_pil = mask_pil.rotate(angle, fillcolor=0)

                if np.random.rand() > 0.5:  # Random Gaussian blur
                    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=np.random.rand() * 2))

                if np.random.rand() > 0.5:  # Random median blur
                    img_np = np.array(img_pil)
                    img_np = cv2.medianBlur(img_np, ksize=5)
                    img_pil = Image.fromarray(img_np)

                if np.random.rand() > 0.5:  # Color perturbation
                    enhancer = ImageEnhance.Color(img_pil)
                    img_pil = enhancer.enhance(np.random.uniform(0.5, 1.5))

            # Convert back to numpy array
            augmented_images.append(np.array(img_pil))
            augmented_masks.append(np.array(mask_pil))

        return np.array(augmented_images), np.array(augmented_masks)

    
    def train_loop(self):
        self.device = torch.device(f'cuda:{self.config.get("gpu", 0)}' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        num_epochs = self.config.get('epochs', 20)

        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config['optimizer_kwargs']['lr'],
            betas=tuple(self.config['optimizer_kwargs']['betas']),
            weight_decay=self.config['optimizer_kwargs']['weight_decay']
        )

        # Initialize the scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_dice = 0.0
        

        for fold_index, (train_loader, valid_loader) in enumerate(tqdm(self.fold_loaders)):
            self.best_val_dice_folds = 0.0
            print(f"Training on fold {fold_index + 1}")
            
            # for epoch in tqdm(range(num_epochs)):
            for epoch in range(num_epochs):

                # Training Loop
                self.train_process(train_loader, epoch, num_epochs)

                # Validation Loop  
                self.eval_process(valid_loader, epoch, num_epochs, fold_index+1)

        print("Training complete")
        self.plot_loss_and_acc(self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies)

    def train_process(self, train_loader, epoch, num_epochs):
        self.model.train()
        train_loss = 0.0
        train_outputs = []
        train_targets = []

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            outputs = outputs['Gland-INST']

            # Compute the loss
            # loss = FocalTverskyLoss(masks, outputs)
            loss = btc_loss(masks, outputs, self.device)
            

            loss.backward()
            self.optimizer.step()

            # Track the running loss
            train_loss += loss.item()
            
            train_outputs.append(outputs)
            # print(f'out shape {outputs.shape}    {torch.argmax(outputs, dim=1).shape}')
            train_targets.append(masks)


        train_outputs = torch.cat(train_outputs)
        # print(f'train_outputs  {train_outputs[-1].shape}')
        train_targets = torch.cat(train_targets)

        # Calculate metrics for training
        train_conf_matrix = ConfusionMatrix(train_outputs, train_targets)
        tp, fp, tn, fn = train_conf_matrix.get_matrix()
        train_conf_matrix_list = [tp, fp, tn, fn]
        train_dice = dice(train_outputs, train_targets, confusion_matrix=train_conf_matrix_list)
        train_jaccard = jaccard(train_outputs, train_targets, confusion_matrix=train_conf_matrix_list)
        train_precision = precision(train_outputs, train_targets, confusion_matrix=train_conf_matrix_list)
        train_recall = recall(train_outputs, train_targets, confusion_matrix=train_conf_matrix_list)
        train_accuracy = accuracy(train_outputs, train_targets, confusion_matrix=train_conf_matrix_list)
        train_f1 = fscore(output=train_outputs, target=train_targets, confusion_matrix=train_conf_matrix_list)
        
        # Print training metrics
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Dice: {train_dice:.4f}, Jaccard: {train_jaccard:.4f}, Precision: {train_precision:.4f}, "
              f"Recall: {train_recall:.4f}, F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}")
        
        self.train_losses.append(train_loss/len(train_loader))
        self.train_accuracies.append(train_accuracy)
        
    def eval_process(self, valid_loader, epoch, num_epochs, fold):
        self.model.eval()
        val_loss = 0.0
        val_outputs = []
        val_targets = []

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(valid_loader):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                outputs = outputs['Gland-INST']
                # loss = FocalTverskyLoss(masks , outputs)
                loss = btc_loss(masks, outputs, self.device)
                val_loss += loss.item()
                val_outputs.append(outputs)
                val_targets.append(masks)
        
        val_outputs = torch.cat(val_outputs)
        val_targets = torch.cat(val_targets)

        # Calculate metrics for validation
        val_conf_matrix = ConfusionMatrix(val_outputs, val_targets)
        tp, fp, tn, fn = val_conf_matrix.get_matrix()
        val_conf_matrix_list = [tp, fp, tn, fn]
        val_dice = dice(val_outputs, val_targets, confusion_matrix=val_conf_matrix_list)
        val_jaccard = jaccard(val_outputs, val_targets, confusion_matrix=val_conf_matrix_list)
        val_precision = precision(val_outputs, val_targets, confusion_matrix=val_conf_matrix_list)
        val_recall = recall(val_outputs, val_targets, confusion_matrix=val_conf_matrix_list)
        val_accuracy = accuracy(val_outputs, val_targets, confusion_matrix=val_conf_matrix_list)
        val_f1 = fscore(output=val_outputs, target=val_targets, confusion_matrix=val_conf_matrix_list)
        
        # Print validation metrics
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss/len(valid_loader):.4f}, "
              f"Dice: {val_dice:.4f}, Jaccard: {val_jaccard:.4f}, Precision: {val_precision:.4f}, "
              f"Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}")

        self.scheduler.step(val_loss / len(valid_loader))
        # Save the best model based on validation loss
        if val_dice > self.best_val_dice:
            self.best_val_dice = val_dice
            checkpoint_path = os.path.join(self.config['output_dir'], f'best_model.pth')
            self.save_model(checkpoint_path)
            print(f'Saved best model to {checkpoint_path}')
        
        if val_dice > self.best_val_dice_folds:
            self.best_val_dice_folds = val_dice
            checkpoint_path = os.path.join(self.config['output_dir'], f'best_model_fold_{fold}.pth')
            self.save_model(checkpoint_path)
            print(f'Saved best fold {fold} model to {checkpoint_path}')
        

        self.val_losses.append(val_loss/len(valid_loader))
        self.val_accuracies.append(val_accuracy)


        
    def test_loop(self):
        self.device = torch.device(f'cuda:{self.config.get("gpu", 0)}' if torch.cuda.is_available() else 'cpu')
        checkpoint_path = self.config['eval_weights_path']
        self.model_args = self.config['model_kwargs']
        self.model = create_model(**self.model_args)
        # print(torch.load(checkpoint_path).keys())
        self.model.load_state_dict(torch.load(checkpoint_path))

        
        self.model.eval()

        self.dataset_module = DatasetModule(
            patch_size = self.config['dataset_kwargs']['input_shape'],
            overlap = 0,
            image_dir = self.config['dataset_kwargs']['test_input_dir'],
            mask_dir = self.config['dataset_kwargs']['test_mask_dir']
        )
        
        patches, masks = self.dataset_module.process_file_list()
        test_dataset = CustomDataset(patches, masks) 
        test_loader = DataLoader(test_dataset, 
                                batch_size=self.config['loader_kwargs']['valid']['batch_size'], 
                                num_workers=self.config['loader_kwargs']['valid']['nr_procs']
                               )

        val_loss = 0.0
        val_outputs = []
        val_targets = []
        j = 0
        self.model.to(self.device)
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(test_loader):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                outputs = outputs['Gland-INST']
                # loss = FocalTverskyLoss(masks , outputs)
                loss = btc_loss(masks, outputs, self.device)
                val_loss += loss.item()
                val_outputs.append(outputs)
                val_targets.append(masks)
                predictions = torch.argmax(outputs, dim=1)
                # for img, msk, pred in zip(images, masks, predictions):
                for i, (msk, pred) in enumerate(zip(masks, predictions)):    
                    # self.visualize_results(img.cpu().numpy(), msk.cpu().numpy(), pred.cpu().numpy())
                    img = np.array(patches[j + batch_idx + i], dtype='uint8')

                    self.visualize_results(img, msk.cpu().numpy(), pred.cpu().numpy())
                j += 1
        
        val_outputs = torch.cat(val_outputs)
        val_targets = torch.cat(val_targets)

        # Calculate metrics for validation
        val_conf_matrix = ConfusionMatrix(val_outputs, val_targets)
        tp, fp, tn, fn = val_conf_matrix.get_matrix()
        val_conf_matrix_list = [tp, fp, tn, fn]
        val_dice = dice(val_outputs, val_targets, confusion_matrix=val_conf_matrix_list)
        val_jaccard = jaccard(val_outputs, val_targets, confusion_matrix=val_conf_matrix_list)
        val_precision = precision(val_outputs, val_targets, confusion_matrix=val_conf_matrix_list)
        val_recall = recall(val_outputs, val_targets, confusion_matrix=val_conf_matrix_list)
        val_accuracy = accuracy(val_outputs, val_targets, confusion_matrix=val_conf_matrix_list)
        val_f1 = fscore(output=val_outputs, target=val_targets, confusion_matrix=val_conf_matrix_list)
        
        # Print test metrics
        print(f"Test Loss: {val_loss/len(test_loader):.4f}, "
              f"Dice: {val_dice:.4f}, Jaccard: {val_jaccard:.4f}, Precision: {val_precision:.4f}, "
              f"Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}")

        # Visualize the results
        # self.visualize_results(images_to_display, masks_to_display, preds_to_display)

    def visualize_results(self, images, masks, preds):
         
        plt.figure(figsize=(12, 10))

        # Original image
        plt.subplot(2, 2, 1)
        # plt.imshow(images.transpose(1, 2, 0))  # Convert from CHW to HWC
        plt.imshow(images) 
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(self.overlay_mask_on_image(images, preds))
        plt.title('Overlay')
        plt.axis('off')
        
        # Ground truth mask
        plt.subplot(2, 2, 3)
        plt.imshow(masks, cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')

        # Predicted mask
        plt.subplot(2, 2, 4)
        plt.imshow(preds, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
        


        plt.show()
        
    def save_results(self, images, masks, preds, epoch, folder):
        folder += "/result_output"
        print(folder)
        if not os.path.exists(folder):
            os.makedirs(folder)

        # Save original image
        image_pil = Image.fromarray((images.transpose(1, 2, 0) * 255).astype('uint8'))  # Convert to HWC and uint8
        image_pil.save(os.path.join(folder, f'{epoch}_image.png'))

        # Save ground truth mask
        mask_pil = Image.fromarray((masks * 255).astype('uint8'))  # Convert mask to uint8
        mask_pil.save(os.path.join(folder, f'{epoch}_mask.png'))

        # Save predicted mask
        pred_pil = Image.fromarray((preds * 255).astype('uint8'))  # Convert prediction to uint8
        pred_pil.save(os.path.join(folder, f'{epoch}_pred.png'))
        
        
    def overlay_mask_on_image(self, image, mask, alpha=0.5):

        # overlayed_image = image.copy()
        # overlayed_image[0] = (1 - alpha) * image[0] + alpha * mask 

        mask[mask == 1] = 255
        t_lower = 30
        t_upper = 100
        # image = np.array(image, dtype='uint8').transpose(1, 2, 0)
        # print(image.shape)
        edges = cv2.Canny(np.array(mask, dtype='uint8'), t_lower, t_upper)
        label = np.zeros_like(image)
        label[edges == 255, :] = [255, 0, 0]
        alpha = 0.6
        beta = 1.0 - alpha
        overlayed_image = np.uint8(alpha * label + beta * image)

        return overlayed_image

    def plot_loss_and_acc(self, train_losses, val_losses, train_accuracies, val_accuracies):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='orange')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training Accuracy', color='blue')
        plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()




