import os
import skimage
import skimage.io
import numpy as np

from connected_components_patches import *


import matplotlib
import matplotlib.pyplot as plt


def image_to_vector(im):
    if im.max() > 1:
        im = im / 255.
    return 1. - im.flatten()



def report_image_almaqrizi_p(model, image_file, output_binary=False):
    # reading colored input image
    image_color = skimage.io.imread(image_file, as_grey=False)

    print '---------------------------------------------------------------------------------------------'
    print 'Image file: "{}"'.format(image_file.split('/')[-1])
    
    # extract patches with their bounding boxes
    bbox_patch_list = list(extract_patches(
        input_file=image_file,
        output_images_height=20,
        output_images_width=20,
        regions_filter=RegionsFilter(),
        region_resizer=MultiplyLowerThanMedianRegionResizer(),
        output_binary=output_binary,
        return_bbox=True
    ))
    bbox_list = [bp[0] for bp in bbox_patch_list]
    patch_list = [bp[1] for bp in bbox_patch_list]

    # get patches al-maqrizi authorship probability prediction
    X_test = [image_to_vector(patch) for patch in patch_list]
    y_test_pred_p = model.predict_proba(X_test)[:,1]
    
    image_p = np.zeros((image_color.shape[:2]))

    for bbox, p in zip(bbox_list, y_test_pred_p):
        image_p[bbox[0]:bbox[2], bbox[1]:bbox[3]] = p
        
    print 'Min p: {0:.3f}; Max p: {1:.3f}; P > 0.5 ration: {2:.3f}.'.format(
        y_test_pred_p.min(),
        y_test_pred_p.max(),
        (y_test_pred_p > 0.5).sum() * 1. / len(y_test_pred_p)
    )
    
    plt.subplot(221)
    plt.imshow(image_color)
    
    plt.subplot(222)
    imgplot = plt.imshow(image_p, cmap='hot')
    plt.colorbar()
    plt.show()
    


import sklearn
import sklearn.neighbors
import sklearn.metrics
import sklearn.linear_model


def buil_model_lr(X_train, y_train, X_valid, y_valid, model=sklearn.linear_model.LogisticRegression()):
    model.fit(X_train, y_train)

    y_valid_pred = model.predict(X_valid)
    y_valid_pred_p = model.predict_proba(X_valid)[:,1]

    y_train_pred = model.predict(X_train)
    y_train_pred_p = model.predict_proba(X_train)[:,1]

    print 'Train:'
    print sklearn.metrics.classification_report(y_train, y_train_pred)
    print 'AUC', sklearn.metrics.roc_auc_score(y_train, y_train_pred_p)

    print '\n'

    print 'Valid:'
    print sklearn.metrics.classification_report(y_valid, y_valid_pred)
    print 'AUC', sklearn.metrics.roc_auc_score(y_valid, y_valid_pred_p)

    return model


def read_images_flat(dir_name):
    for im_file in os.listdir(dir_name):
        im = skimage.io.imread(os.path.join(dir_name, im_file), as_grey=False)
        yield image_to_vector(im)
    

def read_Xy_train_valid(patches_dir):

    X_train_0 = list(read_images_flat(patches_dir + '/train/0'))
    X_train_1 = list(read_images_flat(patches_dir + '/train/1'))

    X_train = X_train_0 + X_train_1
    y_train = [0] * len(X_train_0) + [1] * len(X_train_1)

    X_valid_0 = list(read_images_flat(patches_dir + '/val/0'))
    X_valid_1 = list(read_images_flat(patches_dir + '/val/1'))

    X_valid = X_valid_0 + X_valid_1
    y_valid = [0] * len(X_valid_0) + [1] * len(X_valid_1)

    return X_train, y_train, X_valid, y_valid