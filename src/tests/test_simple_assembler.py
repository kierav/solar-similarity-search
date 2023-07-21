'''
    Unit tests for database class
'''
import sys,os
sys.path.append(os.getcwd())

import unittest
import glob
from datetime import datetime
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from src.simple_assembler import Assembler
import torch

class SimpleAssemblerTest(unittest.TestCase):
    '''
        Test the data_loader class.
    '''
    def setUp(self):
        '''
            Setup the test environment.
        '''
        self.run = 'l0gz0vw6'
        self.embedding_dim = 16
        self.tile_files = np.array(pd.read_csv(sorted(glob.glob('wandb/*'+self.run+'/files/filenamestrain.csv'))[-1])['filename'])
        self.embeddings = np.load(sorted(glob.glob('wandb/*'+self.run+'/files/embeddingstrain.npy'))[-1])
        self.data_path = 'data/test_selectembeddings'
        self.nsave=6
        self.assembler = Assembler(nsave=self.nsave,
                                   run=self.run,data_path=self.data_path)
        

    def test_assemblerExists(self):
        self.assertIsInstance(self.assembler.tile_dim,int)
        self.assertIsInstance(self.assembler.run,str)
        self.assertIsInstance(self.assembler.data_path,str)
        self.assertIsInstance(self.assembler.files,list)
        self.assertEqual(len(self.assembler.files),len(self.assembler.embedding_files))
        self.assertEqual(len(self.assembler.files),len(self.assembler.embeddings_proj_files))

    def test_selectEmbeddings(self):
        sample_files = self.tile_files[:50]
        sample_embeddings = self.embeddings[:50]
        median = np.median(self.embeddings,axis=0)
        print(np.shape(median))
        select_files,select_embeddings = self.assembler.select_interesting_embeddings(sample_files,sample_embeddings,median)
        self.assertEqual(np.shape(select_embeddings),(self.nsave,np.shape(self.embeddings)[1]))
        self.assertEqual(len(select_files),self.nsave)

    def test_selectEmbeddings_toofew(self):
        sample_files = self.tile_files[:5]
        sample_embeddings = self.embeddings[:5]
        median = np.median(self.embeddings,axis=0)
        select_files,select_embeddings = self.assembler.select_interesting_embeddings(sample_files,sample_embeddings,median)
        self.assertEqual(np.shape(select_embeddings),(self.nsave,np.shape(self.embeddings)[1]))
        self.assertEqual(len(select_files),self.nsave)

    def test_createDf(self):
        self.assembler.create_df()
        self.assertIsInstance(self.assembler.df,pd.DataFrame)
        self.assertGreater(len(self.assembler.df),1)
        self.assertIsInstance(self.assembler.df['parent'].iloc[0],str)
        self.assertEqual(len(self.assembler.df),len(self.assembler.embeddings))

    def test_assembleAll(self):
        self.assembler.create_df()
        self.assembler.assemble_all()
        files = os.listdir(self.data_path)
        assembled_file = np.load(self.data_path+os.sep+files[0])
        self.assertEqual(np.shape(assembled_file),(len(self.assembler.df),self.nsave*512))
        self.assertEqual(len(os.listdir(self.data_path)),2)
        self.assertGreater(len(self.assembler.df),len(self.assembler.embeddings)/self.assembler.tile_dim/self.assembler.tile_dim)

    def test_saveDf(self):
        self.assembler.create_df()
        self.assembler.assemble_all()    
        self.assembler.save_df('data','selectembeddings')
        df = pd.read_csv('data'+os.sep+'index_'+self.run+'_selectembeddings.csv')
        self.assertIsInstance(df,pd.DataFrame)
        print(df.head())

if __name__ == "__main__":
    unittest.main()
