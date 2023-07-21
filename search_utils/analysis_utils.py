import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams as rcp
import matplotlib.offsetbox as osb   
import numpy as np
import os
import torch
import torchvision.transforms.functional as functional
from sklearn.metrics.pairwise import cosine_similarity
from image_utils import read_image


def get_scatter_plot_with_thumbnails(embeddings_2d,filenames,root='../'):
    """
    Creates a scatter plot with image overlays and corresponding 2D histogram.
    
    Parameters:
        embeddings_2d (np array):   nx2 array with 2D embeddings for n samples
        filenames (list):           corresponding image file paths 
        root (str):                 root directory for image files

    Returns:
        fig                         figure handle
    """
    
    # initialize empty figure and add subplot
    fig = plt.figure(figsize=[6,3], layout='constrained', dpi=300)
    ax = fig.add_subplot(121)
    # shuffle images and find out which images to show
    shown_images_idx = []
    shown_images = np.array([[1.0, 1.0]])
    iterator = [i for i in range(embeddings_2d.shape[0])]
    np.random.shuffle(iterator)
    for i in iterator:
        # only show image if it is sufficiently far away from the others
        dist = np.sum((embeddings_2d[i] - shown_images) ** 2, 1)
        if np.min(dist) < 2e-3:
            continue
        shown_images = np.r_[shown_images, [embeddings_2d[i]]]
        shown_images_idx.append(i)

    # plot image overlays
    for idx in shown_images_idx:
        thumbnail_size = int(rcp["figure.figsize"][0] * 2.0)
        try:
            img = np.load(root+filenames[idx])
        except:
            continue
        img = np.expand_dims(img,0)
        img = functional.resize(torch.Tensor(img), thumbnail_size,antialias=True)
        img = np.array(img)[0,:,:]
        im = osb.OffsetImage(img, cmap=plt.cm.gray_r)
        img_box = osb.AnnotationBbox(im,embeddings_2d[idx],pad=0.2)
        ax.add_artist(img_box)
        im.get_children()[0].set_clim(-1000,1000)
    
    # set aspect ratio
    ratio = 1.0 / ax.get_data_ratio()
    ax.set_aspect(ratio, adjustable="box")

    # title
    ax.set_title('Embedding space in 2D')

    # 2D histogram
    ax2 = fig.add_subplot(122)
    histout = np.histogram2d(embeddings_2d[:,0], embeddings_2d[:,1], bins=50)
    ax2.hist2d(embeddings_2d[:,0], embeddings_2d[:,1], bins=100, cmap=plt.cm.magma_r, vmax=np.percentile(histout[0], 99))
    ax2.set_title('Density distribution in 2D')

    return fig

def get_image_as_np_array_with_frame(filename: str, w: int = 5):
    """Returns an image as a numpy array with a black frame of width w."""
    img =  read_image(image_loc=filename, image_format = "npy")
    ny, nx = img.shape
    # create an empty image with padding for the frame
    framed_img = -1000*np.ones((w + ny + w, w + nx + w))
    # put the original image in the middle of the new one
    framed_img[w:-w, w:-w] = img
    return framed_img


def plot_nearest_neighbors_3x3(filenames, embeddings, query, query_type: str, i: int, distType: str, root:str='../'):
    """Plots the example image and its eight nearest neighbors."""
    n_subplots = 9
    # initialize empty figure
    fig = plt.figure()
    fig.suptitle(f"Nearest Neighbor Plot {i + 1} with {distType} distance")
    #
    if query_type == 'filename':
        example_idx = np.where(filenames==query)[0]
        print(query)
        embedding_query = embeddings[example_idx,:]
    else:
        embedding_query = np.reshape(query,(1,np.shape(query)[0]))

    # get distances to the cluster center
    distances = []
    if distType.upper() == "EUCLIDEAN":
        distances = embeddings - embedding_query
        distances = np.power(distances, 2).sum(-1).squeeze()
    elif distType.upper() == "COSINE":
        distances = -1*cosine_similarity(embeddings, embedding_query)
        distances = distances[:, 0]
        # distances = [cosine_similarity(np.array([embeddings[i, :]]), np.array([embeddings[example_idx, :]]))[0][0] for i in range(len(embeddings))]

    # sort indices by distance to the center
    nearest_neighbors = np.argsort(distances)[:n_subplots]
    # show images
    for plot_offset, plot_idx in enumerate(nearest_neighbors):
        ax = fig.add_subplot(3, 3, plot_offset + 1)
        # get the corresponding filename
        fname = root + filenames[plot_idx]
        if plot_offset == 0:
            if query_type == 'filename':
                ax.set_title(f"Query Image")
            else:
                ax.set_title(f"Nearest neighbor")
            plt.imshow(get_image_as_np_array_with_frame(fname),cmap='gray',vmin=-1000,vmax=1000)
        else:
            plt.imshow(read_image(image_loc=fname, image_format = "npy"),cmap='gray',vmin=-1000,vmax=1000)
        # let's disable the axis
        plt.axis("off")


def save_predictions(preds,dir,appendstr:str=''):
    """
    Save predicted files and embeddings
    
    Parameters:
        preds:  output of model predict step (as list of batch predictions)
        dir:    directory for saving
        appendstr: string to save at end of filename
    Returns:
        embeddings (np array):      output of model embed step 
        embeddings_proj (np array): output of model projection head
    """
    file = []
    embeddings = []
    embeddings_proj = []
    for predbatch in preds:
        file.extend(predbatch[0])
        embeddings.extend(np.array(predbatch[1]))
        embeddings_proj.extend(np.array(predbatch[2]))
    embeddings = np.array(embeddings)
    embeddings_proj = np.array(embeddings_proj)

    np.save(dir+os.sep+'embeddings'+appendstr+'.npy',embeddings)
    np.save(dir+os.sep+'embeddings_proj'+appendstr+'.npy',embeddings_proj)
    df = pd.DataFrame({'filename':file})
    df.to_csv(dir+os.sep+'filenames'+appendstr+'.csv',index=False)

    return file, embeddings, embeddings_proj


def load_model(ckpt_path,modelclass,api):
    """
    Load model into wandb run by downloading and initializing weights

    Parameters:
        ckpt_path:  wandb path to download model checkpoint from
        model:      model class
        api:        instance of wandb Api
    Returns:
        model:      Instantiated model class object with loaded weights
    """
    print('Loading model checkpoint from ', ckpt_path)
    artifact = api.artifact(ckpt_path,type='model')
    artifact_dir = artifact.download()
    model = modelclass.load_from_checkpoint(artifact_dir+'/model.ckpt',map_location='cpu')
    return model
