import argparse
import glob
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

class Assembler():
    """
    Takes tile embeddings and reassembles into "image"s of size 
    tile_rows x tile_cols x embedding_dim
    """
    def __init__(self,tile_dim:int=128,img_dim:int=2048,embedding_dim:int=16,reduce:bool=True,
                 data_path:str='data',run:str='',datasets:list=['train','val']):
        self.tile_dim = tile_dim
        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.run = run
        self.reduce = reduce

        self.files = []
        self.embedding_files = []
        self.embeddings_proj_files = []
        for dataset in datasets:
            # self.files.append(sorted(glob.glob('wandb/*'+run+'/files/filenames'+dataset+'.csv'))[-1])
            # self.embedding_files.append(sorted(glob.glob('wandb/*'+run+'/files/embeddings'+dataset+'.npy'))[-1])
            # self.embeddings_proj_files.append(sorted(glob.glob('wandb/*'+run+'/files/embeddings_proj'+dataset+'.npy'))[-1])
            self.files.append(sorted(glob.glob('data/embeddings/*'+run+'/filenames'+dataset+'.csv'))[-1])
            self.embedding_files.append(sorted(glob.glob('data/embeddings/*'+run+'/embeddings'+dataset+'.npy'))[-1])
            self.embeddings_proj_files.append(sorted(glob.glob('data/embeddings/*'+run+'/embeddings_proj'+dataset+'.npy'))[-1])
                
        self.df = pd.DataFrame()
        self.embeddings = []
        self.embeddings_proj = []

    def assemble_tiles(self,files,embeddings,dim:int):
        """
        Take tile files and assembles into an "image" of size 
        tile_rows x tile_cols x embedding_dim

        Parameters:
            files (list):       list of tile filepaths
            embeddings (list):  list of corresponding embeddings 
            dim (int)

        Returns:
            img (np array):     tile_rows x tile_cols x embedding_dim
        """
        img = np.zeros((self.img_dim//self.tile_dim,
                       self.img_dim//self.tile_dim,
                       dim))
        
        for file,embedding in zip(files,embeddings):
            file_split = file.split('/')[-1].strip('.npy').split('_')
            ind_row = int(file_split[1])//self.tile_dim
            ind_col = int(file_split[2])//self.tile_dim
            img[ind_row,ind_col,:] = embedding
                    
        return img
    
    def reshape_embedding(self,embeddings,reducer,scaler,fit=True):
        """
        Reshape embedding to specified dimension either by some dimensionality reducer
        or by padding. If reduce flag is set to false, then embeddings are returned 
        unaltered
        
        Parameters:
            embeddings (array):     embeddings to be reshaped
            reducer (sklearn like): has a fit_transform and a transform method
            scaler (sklearn like): has a fit_transform and a transform method
            fit (bool):             whether or not to fit the reducer

        Returns:
            embeddings (array):     reshaped to n samples x embedding dim 
        """
        if not self.reduce:
            return embeddings
        
        if np.shape(embeddings)[1] > self.embedding_dim and fit:
            # embeddings = scaler.fit_transform(embeddings)
            embeddings = reducer.fit_transform(embeddings)
        elif np.shape(embeddings)[1] > self.embedding_dim and not fit:
            # embeddings = scaler.transform(embeddings)
            embeddings = reducer.transform(embeddings)
        else:
            embeddings = np.pad(embeddings,((0,0),(0,self.embedding_dim-np.shape(embeddings)[1])))
        return embeddings

    def create_df(self):
        """
        Iterate through files and embeddings and compile into one dataframe/list
        """
        pca = PCA(n_components=self.embedding_dim)
        pca_proj = PCA(n_components=self.embedding_dim)
        scaler = MinMaxScaler()
        scaler_proj = MinMaxScaler()

        for files,embeddings_file,embeddings_proj_file in zip(self.files,self.embedding_files,self.embeddings_proj_files):
            file_df = pd.read_csv(files)
            file_df['parent'] = file_df['filename'].str.split('/').str[-3]
            file_df['sample_time'] = file_df['filename'].str.split('/').str[-3].str.rsplit('_',n=1).str[0]

            embeddings = np.load(embeddings_file)
            embeddings_proj = np.load(embeddings_proj_file)

            if len(self.df)==0:
                self.df = file_df
                embeddings = self.reshape_embedding(embeddings,pca,scaler,fit=True)
                embeddings_proj = self.reshape_embedding(embeddings_proj,pca_proj,scaler_proj,fit=True)
                self.embeddings = embeddings
                self.embeddings_proj = embeddings_proj

            else:
                self.df = pd.concat((self.df,file_df))
                embeddings = self.reshape_embedding(embeddings,pca,scaler,fit=False)
                embeddings_proj = self.reshape_embedding(embeddings_proj,pca_proj,scaler_proj,fit=False)
                self.embeddings = np.concatenate((self.embeddings,embeddings),axis=0)
                self.embeddings_proj = np.concatenate((self.embeddings_proj,embeddings_proj),axis=0)

        self.df = self.df.reset_index(drop=True)
        self.df['sample_time'] = pd.to_datetime(self.df['sample_time'],format='%Y%m%d_%H%M%S')

    def assemble_all(self):
        """
        To be run after create_df
        Assembles all tile embeddings, saves to data path and creates new index df
        """

        if len(self.df) == 0:
            print('Error, must run create_df first')
            return
        
        for parent in pd.unique(self.df['parent']):
            inds = self.df['parent']==parent
            files = self.df[inds]['filename']
            embeddings = self.embeddings[inds]
            embeddings_proj = self.embeddings_proj[inds]

            img = self.assemble_tiles(files,embeddings,dim=512)
            img_proj = self.assemble_tiles(files,embeddings_proj,dim=128)

            # save image
            np.save(self.data_path+os.sep+parent+'_embedding.npy',img)
            np.save(self.data_path+os.sep+parent+'_embedding_proj.npy',img_proj)

            self.df.loc[inds,'embedding_file'] = self.data_path+os.sep+parent+'_embedding.npy'
            self.df.loc[inds,'embedding_proj_file'] = self.data_path+os.sep+parent+'_embedding_proj.npy'

        print('Assembled',len(self.df),'tiles')
        self.df = self.df.drop_duplicates(subset='parent')
        print('Into',len(self.df),'embedded parent files')

    def save_df(self,dir,filename:str=''):
        """
        Save dataframe to file with labels
        """
        self.df['dataset'] = self.df['parent'].str.split('_').str[-1]
        self.df.sort_values('sample_time',inplace=True)
        self.df.to_csv(dir+os.sep+'index_'+self.run+'_'+filename+'.csv',index=False)
        
        labels = pd.read_csv('../idea-lab-flare-forecast/Data/labels_all_smoothed.csv')
        labels['sample_time'] = pd.to_datetime(labels['sample_time'])
        labels = labels.merge(self.df,how='inner',on=['sample_time','dataset'])
        labels.to_csv('../idea-lab-flare-forecast/Data/labels_'+self.data_path+'.csv',index=False)

def parse_args(args=None):
    """
    Parses command line arguments to script. Sets up argument for which 
    dataset to index.

    Parameters:
        args (list):    defaults to parsing any command line arguments
    
    Returns:
        parser args:    Namespace from argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--run',
                        type=str,
                        help='wandb run id'
                        )
    parser.add_argument('-s','--datasets',
                        type=str,
                        nargs='*',
                        default=['train','val','pseudotest','test'],
                        help='datasets to assemble, i.e., train, val, etc.'
                        )
    parser.add_argument('-re','--reduce',
                        type=bool,
                        default=False,
                        help='flag whether to reduce the embeddings or not')
    parser.add_argument('-e','--embedding_dim',
                        type=int,
                        default=16,
                        help='size of embedding to save'
                        )
    parser.add_argument('-d','--data_path',
                        type=str,
                        help='path to save assembled files')
    return parser.parse_args(args)



if __name__ == '__main__':
    parser = parse_args()
    assembler = Assembler(embedding_dim=parser.embedding_dim,
                          run=parser.run,
                          datasets=parser.datasets,
                          reduce=parser.reduce,
                          data_path=parser.data_path)
    assembler.create_df()
    assembler.assemble_all()
    assembler.save_df('data','assembledfullembeddings')