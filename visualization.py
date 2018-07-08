from glob import glob
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
plt.set_cmap('Greys')

from prepare_data import prepare_raws, prepare_labels


def visualizer(model, X_test, y_test):
    sizes = 256
    three_channels = False
    names_raw = glob('data/Test_images/raws/*.png')
    names_labels = glob('data/Test_images/masks/*.png')
    names_raw.sort()
    names_labels.sort()

    raws = np.stack([prepare_raws(i, sizes=sizes, three_channels=three_channels) for i in names_raw], 0)
    labels = np.stack([prepare_labels(i, sizes=sizes) for i in names_labels], 0)

    threshold = 0.3
    for x in range(len(raws)-1):

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
        x_tm = raws[x]
        x_tm = np.reshape(x_tm, [sizes, sizes])
        y_tm = labels[x]
        y_tm = np.reshape(y_tm, [sizes, sizes])

        pred_tm = model.predict(raws[x:x + 1])

        pred_tm = np.reshape(pred_tm[0], [sizes, sizes])
        pred_tm[pred_tm > threshold] = 1
        pred_tm[pred_tm < threshold] = 0

        ax1.imshow(x_tm)
        ax2.imshow(y_tm)
        ax3.imshow(pred_tm)

        fig.savefig("examples/ex" + str(x) + ".png")