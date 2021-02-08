# Sample notebook
# https://www.kaggle.com/hirune924/train-dog-breed-identification

# notebook install
# !pip install ../input/pikachu-libraries/timm-0.3.4-py3-none-any.whl
# !pip install ../input/pikachu-libraries/omegaconf-2.0.6-py3-none-any.whl
# import sys
# sys.path.append('../input/pikachu-libraries/pytorch-lightning-1.1.7/pytorch-lightning-1.1.7')

####################
# Import Libraries
####################
import os
import sys
import cv2
import numpy as np
import pandas as pd
sys.path.append('../input/pikachu-libraries/pytorch-lightning-1.1.7/pytorch-lightning-1.1.7')


import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning import loggers
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
import albumentations as A
import timm
from omegaconf import OmegaConf

####################
# Config
####################

conf_dict = {'batch_size': 32, 
             'epoch': 1,
             'model_name': 'tf_efficientnet_b0'}
conf_base = OmegaConf.create(conf_dict)


####################
# Dataset
####################

class DogBreedDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None, test=False):
        self.data = dataframe.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.test = test
        cv2.setNumThreads(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.data.loc[idx, "id"] + "." + "jpg")
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)
        image = torch.from_numpy(image["image"].transpose(2, 0, 1))
        if not self.test:
            label = self.data.loc[idx, "label"]
            return image, label
        else:
            return image
           
####################
# Data Module
####################

class DogBreedDataModule(pl.LightningDataModule):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        

    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage=None):
        if stage == 'fit':
            df = pd.read_csv('../input/dog-breed-identification/labels.csv')
            
            df['label'] = df.groupby(['breed']).ngroup()
            self.breed2id_dict = df[['breed','label']].sort_values(by='label').set_index('breed').to_dict()['label'] 
            self.id2breed_dict = df[['breed','label']].sort_values(by='label').set_index('label').to_dict()['breed']  

            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
            for fold, (train_index, val_index) in enumerate(kf.split(df.values, df["breed"])):
                df.loc[val_index, "fold"] = int(fold)
            df["fold"] = df["fold"].astype(int)

            train_df = df[df["fold"] != 0]
            valid_df = df[df["fold"] == 0]

            train_transform = A.Compose([
                        A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0), ratio=(1, 1), interpolation=1, always_apply=False, p=1.0),
                        A.Flip(always_apply=False, p=0.5),
                        A.RandomGridShuffle(grid=(4, 4), always_apply=False, p=1.0),
                        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, always_apply=False, p=0.5),
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
                        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5),
                        A.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0, rotate_limit=45, interpolation=1, border_mode=4, value=255, mask_value=None, always_apply=False, p=0.5),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                        A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
                        ])

            valid_transform = A.Compose([
                        A.Resize(height=256, width=256, interpolation=1, always_apply=False, p=1.0),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                        ])

            self.train_dataset = DogBreedDataset(train_df, '../input/dog-breed-identification/train', transform=train_transform)
            self.valid_dataset = DogBreedDataset(valid_df, '../input/dog-breed-identification/train', transform=valid_transform)
            
        elif stage == 'test':
            test_df = pd.read_csv('../input/dog-breed-identification/sample_submission.csv')
            test_transform = A.Compose([
                        A.Resize(height=256, width=256, interpolation=1, always_apply=False, p=1.0),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                        ])            
            self.test_dataset = DogBreedDataset(test_df, '../input/dog-breed-identification/test', transform=test_transform, test=True)
        

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
        self.model = timm.create_model(model_name=self.hparams.model_name, num_classes=120, pretrained=False, in_chans=3)
        self.criteria = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

    def forward(self, x):
        # use forward for inference/predictions
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
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
        y_hat = torch.cat([x["y_hat"] for x in outputs]).cpu()

        #preds = np.argmax(y_hat, axis=1)

        val_accuracy = self.accuracy(y_hat, y)

        self.log('avg_val_loss', avg_val_loss)
        self.log('val_acc', val_accuracy)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criteria(y_hat, y)
        self.log('test_loss', loss)
        
####################
# Train
####################  
def main():
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf_base, conf_cli)
    print(OmegaConf.to_yaml(conf))
    seed_everything(2021)

    tb_logger = loggers.TensorBoardLogger(save_dir='tb_log/')
    csv_logger = loggers.CSVLogger(save_dir='csv_log/')

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath='ckpt', monitor='avg_val_loss', 
                                          save_last=True, save_top_k=5, mode='min', 
                                          save_weights_only=True, filename='{epoch}-{avg_val_loss:.2f}')

    data_module = DogBreedDataModule(conf)

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
