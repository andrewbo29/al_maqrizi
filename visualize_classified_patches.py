import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage

plt.style.use('ggplot')

def bbox_height(bbox):
    return bbox[3] - bbox[1]


def bbox_width(bbox):
    return bbox[2] - bbox[0]


def plot_image_estimated_al_maqrizi_probability2(
        image_color,
        patch_bbox_p_list,
        cmap=plt.get_cmap('OrRd'),
        font={
            'family': 'Times New Roman',
            'weight': 'normal',
            'size': 18
        },
        nbins=40
):
    """
    Plot source image with classified patches probabilities & some auxiliary plots

    :param image_color: source RGB image
    :param patch_bbox_p_list: patches bboxes and estimated probabilities 
    :type patch_bbox_p_list: list of tuples (bbox, float) 
        where bbox is a skimage-like bounding box: (min_row, min_col, max_row, max_col)

    """
    image_p = np.zeros((image_color.shape[0], image_color.shape[1], 3))
    for bbox, p in patch_bbox_p_list:
        col = cmap(p)
        for i in range(3):
            image_p[bbox[0]:bbox[2], bbox[1]:bbox[3], i] = col[i]

    bbox_list = [bp[0] for bp in patch_bbox_p_list]
    sigma = np.sqrt(np.mean([np.median(map(bbox_height, bbox_list)), np.median(map(bbox_width, bbox_list))]))
    image_p = skimage.filters.gaussian_filter(image_p, sigma=sigma, multichannel=True)

    # resulting image.
    image_color = skimage.exposure.equalize_hist(image_color)
    res_im = (image_color * 0.6 + image_p * 0.4)

    probabilities = np.array([p for bbox, p in patch_bbox_p_list])

    # plotting

    matplotlib.rc('font', **font)

    ax1 = plt.subplot2grid((100, 1), (0, 0), rowspan=78)
    ax2 = plt.subplot2grid((100, 1), (80, 0), rowspan=2)
    ax3 = plt.subplot2grid((100, 1), (85, 0), rowspan=15)

    # plot image itself
    ax1.set_title(
            'Estimated Al-Maqrizi autorship probability'
            + '\n Minimum: {0:.2}, Maximum: {1:.2}, Average: {2:.2}'.format(
                    probabilities.min(), probabilities.max(), probabilities.mean()
            ))
    ax1.tick_params(axis='x', which='both', labelbottom='off')
    ax1.tick_params(axis='y', which='both', labelleft='off')
    ax1.grid(b=False)
    ax1.imshow(res_im)

    # plot color map
    norm = matplotlib.colors.Normalize(0,1)
    cb1 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, orientation='horizontal', ticks=[0, 1], norm=norm)

    # plot probabilities histogram
    n, bins, patches = ax3.hist(probabilities, nbins, normed=0, color='green', range=[0, 1])
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    ## scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    ## setting bin color
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cmap(c))

    ax3.set_yticks([0, 100])
    ax3.set_xticks([0, 1])

    plt.show()

