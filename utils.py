import numpy as np

def prepro(img, down_sample_factor=2):
    img = img[150:300, 150:450] # crop
    img = img[::down_sample_factor, ::down_sample_factor, :] # downsample image
    img = np.mean(img, axis=2) # remove color
    if np.max(img) > 1:
    	img /= 255. # scale between 0 and 1
    return img.astype(np.float)
