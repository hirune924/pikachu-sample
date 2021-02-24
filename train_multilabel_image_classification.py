####################
# Import Libraries
####################
import os
import sys
import cv2
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning import loggers
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
from torch.utils.data import Dataset, DataLoader

#from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
import albumentations as A
import timm
from omegaconf import OmegaConf

####################
# Config
####################

conf_dict = {'batch_size': 32, 
             'epoch': 10,
             'image_size': 512,
             'model_name': 'tf_efficientnet_b0',
             'data_dir': '../input/ranzcr-clip-catheter-line-classification',
             'output_dir': './'}
conf_base = OmegaConf.create(conf_dict)


####################
# Dataset
####################

class RANZCRDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None, train=True):
        self.data = dataframe.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        cv2.setNumThreads(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, "StudyInstanceUID"]
        img_path = os.path.join(self.data_dir, img_name + "." + "jpg")
        
        #img =  np.asarray(Image.open(img_path).convert("RGB"))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            image = self.transform(image=img)
            image = torch.from_numpy(image["image"].transpose(2, 0, 1))

        if self.train:
            label = self.data[self.data["StudyInstanceUID"] == img_name].values.tolist()[0][1:-1]
            label = torch.tensor(label,dtype= torch.float32) 
            return image, label
        else:
            return image
           
####################
# Data Module
####################

class RANZCRDataModule(pl.LightningDataModule):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        

    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage=None):
        if stage == 'fit':
            df = pd.read_csv(os.path.join(self.conf.data_dir, "train.csv"))
            
            train_df, valid_df = model_selection.train_test_split(df, test_size=0.2, random_state=42)

            train_transform = A.Compose([
                        A.RandomResizedCrop(height=512, width=512, scale=(0.9, 1.0), ratio=(1, 1), interpolation=1, always_apply=False, p=1.0),
                        #A.Flip(always_apply=False, p=0.5),
                        #A.RandomGridShuffle(grid=(4, 4), always_apply=False, p=1.0),
                        #A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, always_apply=False, p=0.5),
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
                        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5),
                        #A.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                        #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0, rotate_limit=45, interpolation=1, border_mode=4, value=255, mask_value=None, always_apply=False, p=0.5),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                        A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
                        ])

            valid_transform = A.Compose([
                        A.Resize(height=512, width=512, interpolation=1, always_apply=False, p=1.0),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                        ])

            self.train_dataset = RANZCRDataset(train_df, os.path.join(self.conf.data_dir, 'train'), transform=train_transform)
            self.valid_dataset = RANZCRDataset(valid_df, os.path.join(self.conf.data_dir, 'train'), transform=valid_transform)
            
        elif stage == 'test':
            test_df = pd.read_csv(os.path.join(self.conf.data_dir, "sample_submission.csv"))
            test_transform = A.Compose([
                        A.Resize(height=512, width=512, interpolation=1, always_apply=False, p=1.0),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                        ])
            self.test_dataset = RANZCRDataset(test_df, os.path.join(self.conf.data_dir, 'test'), transform=test_transform, train=False)
         
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)
        
####################
# Lightning Module
####################

class LitSystem(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        #self.conf = conf
        self.save_hyperparameters(conf)
        self.model = timm.create_model(model_name=self.hparams.model_name, num_classes=11, pretrained=True, in_chans=3)
        self.criteria = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        # use forward for inference/predictions
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criteria(y_hat, y)
        
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criteria(y_hat, y)
        
        return {
            "val_loss": loss,
            "y": y,
            "y_hat": y_hat
            }
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = torch.cat([x["y"] for x in outputs]).cpu()
        #y_hat = torch.sigmoid(torch.cat([x["y_hat"] for x in outputs])).cpu()

        #preds = np.argmax(y_hat, axis=1)

        #val_accuracy, _ = get_score(y, y_hat)

        self.log('avg_val_loss', avg_val_loss)
        #self.log('val_acc', val_accuracy)
        
####################
# Train
####################  
def main():
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf_base, conf_cli)
    print(OmegaConf.to_yaml(conf))
    seed_everything(2021)

    tb_logger = loggers.TensorBoardLogger(save_dir=os.path.join(conf.output_dir, 'tb_log/'))
    csv_logger = loggers.CSVLogger(save_dir=os.path.join(conf.output_dir, 'csv_log/'))

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(conf.output_dir, 'ckpt/'), monitor='avg_val_loss', 
                                          save_last=True, save_top_k=5, mode='min', 
                                          save_weights_only=True, filename='{epoch}-{avg_val_loss:.2f}')

    data_module = RANZCRDataModule(conf)

    lit_model = LitSystem(conf)

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=conf.epoch,
        gpus=-1,
        amp_backend='native',
        amp_level='O2',
        precision=16
            )

    trainer.fit(lit_model, data_module)

if __name__ == "__main__":
    main()
