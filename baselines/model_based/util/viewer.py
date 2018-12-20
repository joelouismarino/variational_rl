import numpy as np
import matplotlib.pyplot as plt
import time
plt.ion()

class Viewer(object):
    """
    Creates a matplotlib window to view an incoming stream of images.
    """
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.im = None

    def _preprocess_img(self, img):
        if type(img) != np.ndarray:
            # convert from torch
            img = img.detach().cpu().numpy()
        if len(img.shape) == 4:
            # remove batch dimension
            img = img[0]
        if img.shape[0] == 3:
            # convert to H x W x C
            img = np.transpose(img, (1,2,0))
        return img

    def view(self, img):
        img = self._preprocess_img(img)
        if self.im is None:
            self.im = self.ax.imshow(img)
        else:
            self.im.set_data(img)
        self.fig.canvas.draw()
        time.sleep(0.001)
