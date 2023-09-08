import sys,os
sys.path.append(os.getcwd())

import numpy as np
import torchvision
from torch import nn
import wandb
from sklearn import random_projection
from sklearn.preprocessing import MinMaxScaler, normalize
from src.data import SharpsDataModule
from src.model import BYOL, SimSiam, NNCLR
from search_utils.analysis_utils import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
import torch
import yaml

torchvision.disable_beta_transforms_warning()

def main():    
    # read in config file
    with open('experiment_config.yml') as config_file:
        config = yaml.safe_load(config_file.read())
    

    run = wandb.init(config=config,project=config['meta']['project'],
                        name=config['meta']['name'],
                        group=config['meta']['group'],
                        tags=config['meta']['tags'],
                        )
    config = wandb.config

    # set seeds
    pl.seed_everything(42,workers=True)
    torch.set_float32_matmul_precision('high')

    # define data module
    data = SharpsDataModule(data_file=config.data['data_path'],
                           batch=config.training['batch_size'],
                           augmentation=config.data['augmentation'],
                           normalize=config.model['pretrain'],
                           maxval=config.data['maxval'],
                           dim=config.data['dim'])

    # initialize model
    model = BYOL(lr=config.training['lr'],
                 wd=config.training['wd'],
                 input_channels=config.model['channels'],
                 projection_size=config.model['projection_size'],
                 prediction_size=config.model['prediction_size'],
                 cosine_scheduler_start=config.training['momentum_start'],
                 cosine_scheduler_end=config.training['momentum_end'],
                 loss=config.training['loss'],
                 epochs=config.training['epochs'],
                 pretrain=config.model['pretrain'])

    # model = NNCLR(lr=config.training['lr'],
    #              wd=config.training['wd'],
    #              input_channels=config.model['channels'],
    #              projection_size=config.model['projection_size'],
    #              prediction_size=config.model['prediction_size'],
    #              epochs=config.training['epochs'],
    #              pretrain=config.model['pretrain'])

    
    # initialize wandb logger
    wandb_logger = WandbLogger(log_model='all')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          mode='min',
                                          save_top_k=1,
                                          save_last=True,
                                          save_weights_only=True,
                                          verbose=False)

    # train model
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         max_epochs=config.training['epochs'],
                         log_every_n_steps=50,
                        #  limit_train_batches=1000,
                         limit_val_batches=400,
                         logger=wandb_logger,
                         deterministic=True,
                         precision='16-mixed')
    trainer.fit(model=model,datamodule=data)

    # save predictions for training
    api = wandb.Api()
    model = load_model('kierav/'+config.meta['project']+'/model-'+run.id+':latest', model, api)

    # local save directory
    savedir = 'data/embeddings/run-'+run.id
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        
    preds_train = trainer.predict(model=model,dataloaders=data.train_dataloader(shuffle=False))
    files_train, embeddings_train,embeddings_proj_train = save_predictions(preds_train,savedir,'train')

    # save predictions for validation
    preds_val = trainer.predict(model=model,dataloaders=data.val_dataloader())
    files_val, embeddings_val,embeddings_proj_val = save_predictions(preds_val,savedir,'val')

    preds_test = trainer.predict(model=model,dataloaders=data.test_dataloader())
    files_test, embeddings_test,embeddings_proj_test = save_predictions(preds_test,savedir,'test')


    wandb.finish()

if __name__ == "__main__":
    main()
