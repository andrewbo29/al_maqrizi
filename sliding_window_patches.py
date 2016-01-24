import numpy as np
import skimage.io
import skimage.filters
import os


def get_binary_map(image_name):
    image = skimage.io.imread(image_name, as_grey=True)

    global_thresh = skimage.filters.threshold_otsu(image)
    binary_global = image > global_thresh

    return np.where(binary_global, image, 0.)


def is_valid_patch(patch):
    height = patch.shape[0]
    weight = patch.shape[1]

    threshold = 0.9

    for i in range(height):
        res = float((patch[i, :] == 0).sum()) / weight
        if res >= threshold:
            return False

    for j in range(weight):
        res = float((patch[:, j] == 0).sum()) / height
        if res >= threshold:
            return False

    return True


def get_sliding_window_patches(image_name, binary_map, window_size=152, stride=10):
    image = skimage.io.imread(image_name)

    x1 = 0
    y1 = 0
    x2 = window_size
    y2 = window_size

    threshold_down = 0.63
    threshold_up = 0.67

    patches = []
    while x2 < binary_map.shape[0]:
        binary_patch = binary_map[x1:x2, y1:y2]
        if threshold_down < binary_patch.mean() < threshold_up:
            if is_valid_patch(binary_patch):
                patches.append(image[x1:x2, y1:y2])

        y1 += stride
        y2 = y1 + window_size
        if y2 >= binary_map.shape[1]:
            y1 = 0
            y2 = window_size
            x1 += stride
            x2 = x1 + window_size

    return patches


def show_rand_patch(patches_list):
    ind = np.random.choice(len(patches_list))
    skimage.io.imshow(patches_list[ind])
    skimage.io.show()


def show_patches(patches_list):
    for ind in range(len(patches_list)):
        skimage.io.imshow(patches_list[ind])
        skimage.io.show()


def show_patch(patches_list, patch_ind):
    skimage.io.imshow(patches_list[patch_ind])
    skimage.io.show()


def save_patches(patches_list, dir_name, image_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    image_fname = image_name.split('/')[-1][:-4]
    patches_dir_name = '%s/%s' % (dir_name, image_fname)
    os.mkdir(patches_dir_name)

    patch_num = 0
    for patch in patches_list:
        patch_fname = '%s/%s.png' % (patches_dir_name, patch_num)
        skimage.io.imsave(patch_fname, patch)
        patch_num += 1


img_name = '../data/al-maqrizi/Archive_1/Ms-orient-A-01771_014.jpg'

bin_map = get_binary_map(img_name)
crops = get_sliding_window_patches(img_name, bin_map, 152, 50)
print len(crops)

save_patches(crops, '../data', img_name)