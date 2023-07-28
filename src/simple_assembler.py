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
    def __init__(self,tile_dim:int=128,img_dim:int=2048,nsave:int=6,embedding_dim:int=10,
                 data_path:str='data',run:str='',datasets:list=['train','val']):
        self.tile_dim = tile_dim
        self.img_dim = img_dim
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.run = run
        self.n = nsave     # number of embeddings to keep
        self.embedding_dim = embedding_dim  # reduced embedding dimension to save

        self.files = []
        self.embedding_files = []
        self.embeddings_proj_files = []
        for dataset in datasets:
            self.files.append(sorted(glob.glob('wandb/*'+run+'/files/filenames'+dataset+'.csv'))[-1])
            self.embedding_files.append(sorted(glob.glob('wandb/*'+run+'/files/embeddings'+dataset+'.npy'))[-1])
            self.embeddings_proj_files.append(sorted(glob.glob('wandb/*'+run+'/files/embeddings_proj'+dataset+'.npy'))[-1])
        
        self.df = pd.DataFrame()
        self.embeddings = []
        self.embeddings_proj = []
    
    def select_interesting_embeddings(self,files,embeddings,embeddings_reduced,median,median_reduced):
        """
        Choose the embeddings which are furthest away from the median
        
        Parameters:
            files (list):               filenames (length n_samples)
            embeddings (array-like):    n_samples x n_features
            embeddings_reduced (array-like):    n samples x embedding dim
            median (array-like):        1 x n_features, embedding median
            median_reduced (array-like):1 x embedding dim, median 
        Returns:
            select_files (list):            filenames corresponding to selected embeddings (length n)
            select_embeddings (array-like):  length n x n_features, selected embeddings
            select_embeddings_reduced (array-like): length n x embedding dim
        """
        distances = np.linalg.norm(embeddings-median,axis=1)
        idx_furthest = np.argsort(distances)[-self.n:]

        select_embeddings = np.tile(median,(self.n,1))
        select_embeddings[:len(idx_furthest),:] = embeddings[idx_furthest]

        select_embeddings_reduced = np.tile(median_reduced,(self.n,1))
        select_embeddings_reduced[:len(idx_furthest),:] = embeddings_reduced[idx_furthest]

        select_files = self.n*['']
        select_files[:len(idx_furthest)] = np.array(files)[idx_furthest]

        return select_files,select_embeddings,select_embeddings_reduced
    
    def reshape_embedding(self,embeddings,reducer,scaler,fit=True):
        """
        Reshape embedding to specified dimension either by some dimensionality reducer
        or by padding
        
        Parameters:
            embeddings (array):     embeddings to be reshaped
            reducer (sklearn like): has a fit_transform and a transform method
            scaler (sklearn like): has a fit_transform and a transform method
            fit (bool):             whether or not to fit the reducer

        Returns:
            embeddings (array):     reshaped to n samples x embedding dim 
        """
        if np.shape(embeddings)[1] > self.embedding_dim and fit:
            embeddings = scaler.fit_transform(embeddings)
            embeddings = reducer.fit_transform(embeddings)
        elif np.shape(embeddings)[1] > self.embedding_dim and not fit:
            embeddings = scaler.transform(embeddings)
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

        for files,embeddings_file,embeddings_proj_file in zip(self.files,self.embedding_files,self.embeddings_proj_files):
            file_df = pd.read_csv(files)
            file_df['parent'] = file_df['filename'].str.split('/').str[-3]
            file_df['sample_time'] = file_df['filename'].str.split('/').str[-3].str.rsplit('_',n=1).str[0]

            embeddings = np.load(embeddings_file)
            embeddings_proj = np.load(embeddings_proj_file)

            if len(self.df)==0:
                self.df = file_df
                self.median_embedding =  np.median(embeddings,axis=0)
                self.median_embedding_proj = np.median(embeddings_proj,axis=0)
                self.embeddings = embeddings
                self.embeddings_proj = embeddings_proj
                self.embeddings_reduced = pca.fit_transform(embeddings)
                self.embeddings_proj_reduced = pca_proj.fit_transform(embeddings_proj)
                self.median_embedding_reduced = pca.transform(self.median_embedding.reshape(1,-1))
                self.median_embedding_proj_reduced = pca_proj.transform(self.median_embedding_proj.reshape(1,-1))

            else:
                self.df = pd.concat((self.df,file_df))
                self.embeddings = np.concatenate((self.embeddings,embeddings),axis=0)
                self.embeddings_proj = np.concatenate((self.embeddings_proj,embeddings_proj),axis=0)
                self.embeddings_reduced = np.concatenate((self.embeddings_reduced,pca.transform(embeddings)),axis=0)
                self.embeddings_proj_reduced = np.concatenate((self.embeddings_proj_reduced,pca_proj.transform(embeddings_proj)),axis=0)

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
        
        save_embeddings = []
        save_embeddings_proj = []
        save_embeddings_reduced = []
        save_embeddings_proj_reduced = []

        for parent in pd.unique(self.df['parent']):
            inds = self.df['parent']==parent
            files = self.df[inds]['filename']
            embeddings = self.embeddings[inds]
            embeddings_proj = self.embeddings_proj[inds]

            select_files,select_embeddings,select_embeddings_reduced = self.select_interesting_embeddings(files,embeddings,
                                                                                self.embeddings_reduced[inds],
                                                                                median=self.median_embedding,
                                                                                median_reduced=self.median_embedding_reduced)
            select_files_proj,select_embeddings_proj,select_embeddings_proj_reduced = self.select_interesting_embeddings(files,
                                                                                embeddings_proj,
                                                                                self.embeddings_proj_reduced[inds],
                                                                                median=self.median_embedding_proj,
                                                                                median_reduced=self.median_embedding_proj_reduced)
            
            # save flattened interesting embeddings
            save_embeddings.append(select_embeddings.flatten())
            save_embeddings_proj.append(select_embeddings_proj.flatten())
            save_embeddings_reduced.append(select_embeddings_reduced.flatten())
            save_embeddings_proj_reduced.append(select_embeddings_proj_reduced.flatten())
                                        
            for i in range(len(select_files)):
                self.df.loc[inds,'embedding_file_'+str(i)] = select_files[i]
                self.df.loc[inds,'embedding_proj_file_'+str(i)] = select_files_proj[i]

        np.save(self.data_path+os.sep+'embeddings.npy',save_embeddings)
        np.save(self.data_path+os.sep+'embeddings_proj.npy',save_embeddings_proj)
        np.save(self.data_path+os.sep+'embeddings_reduced.npy',save_embeddings_reduced)
        np.save(self.data_path+os.sep+'embeddings_proj_reduced.npy',save_embeddings_proj_reduced)

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
        labels.to_csv('../idea-lab-flare-forecast/Data/labels_selectembeddings_'+self.run+'.csv',index=False)

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
    parser.add_argument('-n','--nsave',
                        type=int,
                        default=6,
                        help='number of embeddings to save'
                        )
    parser.add_argument('-e','--embeddingdim',
                        type=int,
                        default=10,
                        help='reduced embedding dimension to save'
                        )
    parser.add_argument('-d','--data_path',
                        type=str,
                        help='path to solve assembled files')
    return parser.parse_args(args)



if __name__ == '__main__':
    parser = parse_args()
    assembler = Assembler(embedding_dim=parser.embeddingdim,
                          nsave=parser.nsave,run=parser.run,datasets=parser.datasets,data_path=parser.data_path)
    assembler.create_df()
    assembler.assemble_all()
    assembler.save_df('data','selectembeddings')