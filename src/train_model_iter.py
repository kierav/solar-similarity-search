import sys,os
sys.path.append(os.getcwd())

import numpy as np
import torchvision
from torch import nn
import wandb
from sklearn.decomposition import PCA
from sklearn import random_projection
from sklearn.preprocessing import MinMaxScaler, normalize
from src.data import TilesDataModule
from src.model import BYOL, SimSiam, NNCLR
from search_utils.analysis_utils import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
import random
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

    # local save directory
    savedir = 'data/embeddings/run-'+run.id
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # set seeds
    pl.seed_everything(42,workers=True)
    torch.set_float32_matmul_precision('high')

    # define data module
    data = TilesDataModule(data_path=config.data['data_path'],
                           batch=config.training['batch_size'],
                           augmentation=config.data['augmentation'],
                           normalize=config.model['pretrain'])

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
    
    # select initial random subset of training data
    data.prepare_data()
    data.setup(stage='train')
    subset_files = random.sample(data.df_train['filename'].tolist(),k=int(config.training['train_frac']*len(data.df_train)))
    
    # iterate training 
    for i in range(config.training['iterations']):
        # create training dataset from desired files
        data.subsample_trainset(subset_files)

        trainer = pl.Trainer(accelerator='gpu',
                            devices=1,
                            max_epochs=config.training['epochs'],
                            log_every_n_steps=50,
                            limit_val_batches=400,
                            logger=wandb_logger,
                            deterministic=True,
                            precision='16-mixed')
        trainer.fit(model=model,train_dataloaders=data.subset_train_dataloader(),val_dataloaders=data.val_dataloader())

        # run inference on full training data 
        preds_train = trainer.predict(dataloaders=data.train_dataloader(shuffle=False))
        files_train, embeddings_train,embeddings_proj_train = save_predictions(preds_train,savedir,'train')
        
        # select diverse subset based on embedding pca
        pca = PCA(n_components=6,random_state=42)
        embeddings_pca = pca.fit_transform(embeddings_train)
        subset_files,_ = diverse_sampler(files_train,embeddings_pca,n=int(config.training['train_frac']*len(files_train)))


    # save predictions for training
    api = wandb.Api()
    model = load_model('kierav/'+config.meta['project']+'/model-'+run.id+':latest', model, api)
        
    # normalize and project embeddings into 2D for plotting
    projection = random_projection.GaussianRandomProjection(n_components=2)
    embeddings_2d_train = projection.fit_transform(embeddings_train)    
    
    scaler = MinMaxScaler()
    embeddings_2d_train = scaler.fit_transform(embeddings_2d_train)
    fig = get_scatter_plot_with_thumbnails(embeddings_2d_train,files_train,'')
    wandb.log({"Backbone_embeddings_2D_train": wandb.Image(fig)})

    projection2 = random_projection.GaussianRandomProjection(n_components=2)
    embeddings_proj2D_train = projection2.fit_transform(embeddings_proj_train)    

    scaler2 = MinMaxScaler()
    embeddings_proj2D_train = scaler2.fit_transform(embeddings_proj2D_train)
    fig2 = get_scatter_plot_with_thumbnails(embeddings_proj2D_train,files_train,'')
    wandb.log({"Projection_embeddings_2D_train": wandb.Image(fig2)})

    # save predictions for validation
    preds_val = trainer.predict(model=model,dataloaders=data.val_dataloader())
    files_val, embeddings_val,embeddings_proj_val = save_predictions(preds_val,savedir,'val')

    # normalize and project embeddings into 2D for plotting
    embeddings_2d_val = projection.transform(embeddings_val)    
    embeddings_2d_val = scaler.transform(embeddings_2d_val)
    fig3 = get_scatter_plot_with_thumbnails(embeddings_2d_val,files_val,'')
    wandb.log({"Backbone_embeddings_2D_val": wandb.Image(fig3)})

    embeddings_proj2d_val = projection2.transform(embeddings_proj_val)    
    embeddings_proj2d_val = scaler2.transform(embeddings_proj2d_val)
    fig4 = get_scatter_plot_with_thumbnails(embeddings_proj2d_val,files_val,'')
    wandb.log({"Projection_embeddings_2D_val": wandb.Image(fig4)})

    preds_pseudotest = trainer.predict(model=model,dataloaders=data.pseudotest_dataloader())
    files_pt, embeddings_pt,embeddings_proj_pt = save_predictions(preds_pseudotest,savedir,'pseudotest')

    preds_test = trainer.predict(model=model,dataloaders=data.test_dataloader())
    files_test, embeddings_test,embeddings_proj_test = save_predictions(preds_test,wandb.run.dir,'test')


    wandb.finish()

if __name__ == "__main__":
    main()
