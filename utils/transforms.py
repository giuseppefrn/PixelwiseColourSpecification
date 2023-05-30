from skimage.color import rgb2lab, deltaE_ciede2000
import numpy as np 

def deltae_2000(true, pred):
    """
    - x and y are the batch tensors
    """

    # print(true.shape)

    x = np.moveaxis(true.cpu().numpy(), 1, -1)
    y = np.moveaxis(pred.cpu().numpy(), 1, -1)

    x_lab = rgb2lab(x)
    y_lab = rgb2lab(y)

    #calc delta e
    res = np.mean(deltaE_ciede2000(x_lab, y_lab))

    # print(res.shape)
    # print(np.mean(res))

    return res