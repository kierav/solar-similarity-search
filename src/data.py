import sys,os
sys.path.append(os.getcwd())

from datetime import datetime,timedelta
import glob
import torch
import torchvision.transforms.v2 as transforms
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset,DataLoader

def split_data(df,val_split,test=''):
    """
        Split dataset into training, validation, hold-out (pseudotest) and test sets.
        The test set can be either 'test_a' which is all data from November and December,
        or 'test_b' which is >=2021, or both combined. The hold-out set is data from
        Sept 15-Oct 31. The remaining data is split temporally 4:1 into training and 
        validation.

        Parameters:
            df (dataframe):     Pandas dataframe containing all the data
            val_split (0-4):    Number between 0-4 indicating which temporal training/validation split to select    
            test (str):         Which test set to choose ('test_a' or 'test_b', otherwise both)

        Returns:
            df_test (dataframe):        Test set
            df_pseudotest (dataframe):  Hold-out set
            df_train (dataframe):       Training set
            df_val (dataframe):         Validation set
    """

    # hold out test sets
    inds_test_a = (df['sample_time'].dt.month >= 11)
    inds_test_b = (df['sample_time']>=datetime(2021,1,1))
    
    # select test set
    if test == 'test_a':
        inds_test = inds_test_a
    elif test == 'test_b':
        inds_test = inds_test_b
    else:
        inds_test = inds_test_a | inds_test_b

    df_test = df.loc[inds_test,:]
    df_full = df.loc[~inds_test,:]

    # select pseudotest/hold-out set
    if test == 'test_a':
        inds_pseudotest = ((df_full['sample_time'].dt.month==10)&(df_full['sample_time'].dt.day>26)) | ((df_full['sample_time'].dt.month==1)&(df_full['sample_time'].dt.day<6))
    elif test == 'test_b':
        inds_pseudotest = (df['sample_time']>=datetime(2020,12,26))
    else:
        inds_pseudotest = (df_full['sample_time'].dt.month==10) | ((df_full['sample_time'].dt.month==9)&(df_full['sample_time'].dt.day>15))
    # inds_pseudotest = (df['sample_time']<datetime(1996,1,1)) | ((df_full['sample_time'].dt.month==10)&(df_full['sample_time'].dt.day>26)) | ((df_full['sample_time'].dt.month==1)&(df_full['sample_time'].dt.day<6))
    df_pseudotest = df_full.loc[inds_pseudotest,:]

    # split training and validation
    df_train = df_full.loc[~inds_pseudotest,:]
    df_train = df_train.reset_index(drop=True)
    n_val = int(np.floor(len(df_train)/5))
    df_val = df_train.iloc[val_split*n_val:(val_split+1)*n_val,:]
    df_train = df_train.drop(df_val.index)

    return df_test,df_pseudotest,df_train,df_val
class TilesDataset(Dataset):
    """
        Pytorch dataset for handling magnetogram tile data 
        
    """
    def __init__(self, image_files: list, transform, augmentation: str='single', 
                 instrument:str='mag',filetype:str='npy',
                 datatype=np.float32, normalize=False):
        '''
            Initializes image files in the dataset
            
            Args:
                image_files (list): list of file paths to images
                augmentation (str): whether to just return the original patches ('none')
                                    perform single augmentation ('single') 
                                    or perform double augmentation ('double').
                                    No augmentation returns a single image,
                                    single or double augmentation returns two.
                data_stride (int): stride to use when loading the images to work 
                                    with a reduced version of the data
                datatype (numpy.dtype): datatype to use for the images
        '''
        self.image_files = image_files
        self.transform = transform
        self.filetype=filetype
        self.datatype=datatype
        self.augmentation = augmentation
        if self.augmentation is None:
            self.augmentation = 'none'
        self.norm = transforms.Normalize(mean=(0.485,),std=(0.229,))
        self.normalize = normalize

    def __len__(self):
        '''
            Calculates the number of images in the dataset
                
            Returns:
                int: number of images in the dataset
        '''
        return len(self.image_files)

    def __getitem__(self, idx):
        '''
            Retrieves an image from the dataset and creates a copy of it,
            applying a series of random augmentations to the copy.

            Args:
                idx (int): index of the image to retrieve
                
            Returns:
                tuple: (image, image2) where image2 is an augmented modification
                of the input and image can be the original image, or another augmented
                modification, in which case image2 is double augmented
        '''
        file = self.image_files[idx]
        image = read_image(image_loc=file,image_format=self.filetype)
        image = np.nan_to_num(image)
        # Normalize magnetogram data
        # clip magnetogram data within max value
        maxval = 1000  # Gauss
        image[np.where(image>maxval)] = maxval
        image[np.where(image<-maxval)] = -maxval
        # scale between 0 and 1
        image = (image+maxval)/2/maxval
        image = np.expand_dims(image,0)

        image2 = image.copy()
        image = torch.Tensor(image)
        image2 = torch.Tensor(image2)

        if self.augmentation.lower() != 'none':

            # aug = Augmentations(image, self.augmentation_list.randomize())
            # image2, _ = aug.perform_augmentations(fill_void='Nearest')
            image2 = self.transform(image)

            if self.augmentation.lower() == 'double':
                image = self.transform(image)    

        if self.normalize:
            image = self.norm(image)

        return file,image,image2

    
class TilesDataModule(pl.LightningDataModule):
    """
    Datamodule for self supervision on tiles dataset
    """

    def __init__(self,data_path:str,batch:int=128,augmentation:str='double',
                 filetype:str='npy',dim:int=128,val_split:int=0,test:str='',
                 normalize=False):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch
        self.augmentation = augmentation
        self.filetype = filetype
        self.val_split = val_split
        self.test = test
        self.normalize = normalize

        # define data transforms - augmentation for training
        self.transform = transforms.Compose([
            transforms.RandomInvert(p=0.3),
            transforms.RandomAdjustSharpness(1.5,p=0.3),
            transforms.RandomApply(torch.nn.ModuleList([
                 transforms.GaussianBlur(kernel_size=9,sigma=(0.1,4.0)),
                 ]),p=0.3),
            transforms.RandomApply(torch.nn.ModuleList([
                 transforms.RandomRotation(degrees=45,fill=0.5)
                 ]),p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ScaleJitter(target_size=(128,128),scale_range=(0.7,1.3),antialias=True),
            transforms.Resize((dim,dim),antialias=True),
        ])

    def prepare_data(self):
        self.image_files = glob.glob(self.data_path + "/**/*."+self.filetype, recursive=True)
        self.df = pd.DataFrame({'filename':self.image_files})
        self.df['sample_time'] = self.df['filename'].str.split('/').str[-3].str.rsplit('_',n=1).str[0]
        self.df['sample_time'] = pd.to_datetime(self.df['sample_time'],format='%Y%m%d_%H%M%S')
        
    def setup(self,stage:str):
        # split into training and validation
        df_test,df_pseudotest,self.df_train,df_val = split_data(self.df,self.val_split,self.test)
        self.train_set = TilesDataset(self.df_train['filename'].tolist(),self.transform,augmentation=self.augmentation,normalize=self.normalize)
        self.val_set = TilesDataset(df_val['filename'].tolist(),self.transform,augmentation='single',normalize=self.normalize)
        self.pseudotest_set = TilesDataset(df_pseudotest['filename'].tolist(),self.transform,augmentation='none',normalize=self.normalize)
        self.test_set = TilesDataset(df_test['filename'].tolist(),self.transform,augmentation='none',normalize=self.normalize)
        self.trainval_set = TilesDataset(pd.concat([self.df_train,df_val])['filename'].tolist(),self.transform,augmentation='none',normalize=self.normalize)

    def subsample_trainset(self,filenames):
        # given a list of filenames, subsample so the train set only includes files from that list
        subset_filenames = self.df_train['filename'][self.df_train['filename'].isin(filenames)].tolist()
        self.subset_train_set = TilesDataset(subset_filenames,self.transform,augmentation=self.augmentation,normalize=self.normalize)

    def subset_train_dataloader(self,shuffle=True):
        return DataLoader(self.subset_train_set,batch_size=self.batch_size,num_workers=4,shuffle=shuffle)
    
    def train_dataloader(self,shuffle=True):
        return DataLoader(self.train_set,batch_size=self.batch_size,num_workers=4,shuffle=shuffle)
    
    def val_dataloader(self):
        return DataLoader(self.val_set,batch_size=self.batch_size,num_workers=4)
    
    def pseudotest_dataloader(self):
        return DataLoader(self.pseudotest_set,batch_size=self.batch_size,num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_set,batch_size=self.batch_size,num_workers=4)
    
class SharpsDataset(Dataset):
    """
        Pytorch dataset for handling SHARPs magnetograms data 
        
    """
    def __init__(self, df: pd.DataFrame, transform, augmentation: str='single', 
                 datatype=np.float32, normalize=False, maxval:float=1000):
        '''
            Initializes image files in the dataset
            
            Args:
                df (dataframe):     pandas dataframe with file paths and features
                transform:          torchvision transform or list of transforms 
                augmentation (str): whether to just return the original patches ('none')
                                    perform single augmentation ('single') 
                                    or perform double augmentation ('double').
                                    No augmentation returns a single image,
                                    single or double augmentation returns two.
                datatype (numpy.dtype): datatype to use for the images
                normalize (bool):   whether or not to normalize for 1 channel pretrained data
                maxval (float):     value for max scaling (Gauss)
        '''
        self.name_frame = df.loc[:,'file']
        self.transform = transform
        self.datatype=datatype
        self.augmentation = augmentation
        if self.augmentation is None:
            self.augmentation = 'none'
        self.norm = transforms.Normalize(mean=(0.485,),std=(0.229,))
        self.normalize = normalize
        self.maxval = maxval

    def __len__(self):
        '''
            Calculates the number of images in the dataset
                
            Returns:
                int: number of images in the dataset
        '''
        return len(self.name_frame)

    def __getitem__(self, idx):
        '''
            Retrieves an image from the dataset and creates a copy of it,
            applying a series of random augmentations to the copy.

            Args:
                idx (int): index of the image to retrieve
                
            Returns:
                tuple: (image, image2) where image2 is an augmented modification
                of the input and image can be the original image, or another augmented
                modification, in which case image2 is double augmented
        '''
        file = self.name_frame.iloc[idx]
        image = np.array(h5py.File(file,'r')['hmi']).astype(np.float32)
        image = np.nan_to_num(image)

        # Clip and normalize magnetogram data
        image = (np.clip(image,-self.maxval,self.maxval)/self.maxval+1)/2

        image = np.transpose(image,(1,2,0))

        image2 = image.copy()
        image = torch.Tensor(image)
        image2 = torch.Tensor(image2)

        if self.augmentation.lower() != 'none':

            # aug = Augmentations(image, self.augmentation_list.randomize())
            # image2, _ = aug.perform_augmentations(fill_void='Nearest')
            image2 = self.transform(image)

            if self.augmentation.lower() == 'double':
                image = self.transform(image)    

        if self.normalize:
            image = self.norm(image)

        return file,image,image2

    
class SharpsDataModule(pl.LightningDataModule):
    """
    Datamodule for self supervision on Sharps dataset
    """

    def __init__(self,data_file:str,batch:int=128,augmentation:str='double',
                 dim:int=128,val_split:int=0,test:str='',
                 normalize=False,maxval:float=1000):
        super().__init__()
        self.data_file = data_file
        self.batch_size = batch
        self.augmentation = augmentation
        self.val_split = val_split
        self.test = test
        self.normalize = normalize
        self.maxval = maxval

        # define data transforms - augmentation for training
        self.transform = transforms.Compose([
            transforms.RandomInvert(p=0.3),
            transforms.RandomAdjustSharpness(1.5,p=0.3),
            # transforms.RandomApply(torch.nn.ModuleList([
            #      transforms.GaussianBlur(kernel_size=9,sigma=(0.1,4.0)),
            #      ]),p=0.3),
            transforms.RandomApply(torch.nn.ModuleList([
                 transforms.RandomRotation(degrees=45,fill=0.5)
                 ]),p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ScaleJitter(target_size=(128,128),scale_range=(0.7,1.3),antialias=True),
            transforms.Resize((dim,dim),antialias=True),
        ])

    def prepare_data(self):
        self.df = pd.read_csv(self.data_file)
        self.df['sample_time'] = pd.to_datetime(self.df['sample_time'],format='mixed')
        
    def setup(self,stage:str):
        # split into training and validation the same as for forecasting
        df_test,df_pseudotest,self.df_train,df_val = split_data(self.df,self.val_split,self.test)

        # use training+val together and pseudotest as validation
        self.df_train = pd.concat([self.df_train,df_val])

        # create datasets
        self.train_set = SharpsDataset(self.df_train,self.transform,augmentation=self.augmentation,normalize=self.normalize,maxval=self.maxval)
        self.val_set = SharpsDataset(df_pseudotest,self.transform,augmentation='single',normalize=self.normalize,maxval=self.maxval)
        self.test_set = SharpsDataset(df_test,self.transform,self.transform,augmentation='none',normalize=self.normalize,maxval=self.maxval)
        print('Train:',len(self.train_set),
              'Valid:',len(self.val_set),
              'Test:',len(self.test_set))

    def subsample_trainset(self,filenames):
        # given a list of filenames, subsample so the train set only includes files from that list
        subset_df = self.df_train[self.df_train['file'].isin(filenames)]
        self.subset_train_set = SharpsDataset(subset_df,self.transform,augmentation=self.augmentation,normalize=self.normalize,maxval=self.maxval)

    def subset_train_dataloader(self,shuffle=True):
        return DataLoader(self.subset_train_set,batch_size=self.batch_size,num_workers=4,shuffle=shuffle)
    
    def train_dataloader(self,shuffle=True):
        return DataLoader(self.train_set,batch_size=self.batch_size,num_workers=4,shuffle=shuffle)
    
    def val_dataloader(self):
        return DataLoader(self.val_set,batch_size=self.batch_size,num_workers=4)
    
    def pseudotest_dataloader(self):
        return DataLoader(self.pseudotest_set,batch_size=self.batch_size,num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_set,batch_size=self.batch_size,num_workers=4)