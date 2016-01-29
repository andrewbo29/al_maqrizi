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


def save_patches(patches_list, dir_name, txt_fname, image_name, label):
    with open(txt_fname, 'a') as txt_file:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        image_fname = image_name.split('/')[-1][:-4]
        patches_dir_name = '%s/%s' % (dir_name, image_fname)
        if not os.path.exists(patches_dir_name):
            os.mkdir(patches_dir_name)

        patch_num = 0
        for patch in patches_list:
            patch_fname = '%s/%s.png' % (patches_dir_name, patch_num)
            skimage.io.imsave(patch_fname, patch)
            patch_num += 1
            txt_file.write('%s %d\n' % (patch_fname, label))


def process_image(image_name, window_size, stride, dir_name, txt_fname, label):
    print 'Processing image %s' % image_name

    bin_map = get_binary_map(image_name)
    crops = get_sliding_window_patches(image_name, bin_map, window_size, stride)
    save_patches(crops, dir_name, txt_fname, image_name, label)

    print 'Number of patches: %d' % len(crops)


def process_images_path(path, window_size, stride, dir_name, txt_fname, label):
    for image_name in os.listdir(path):
        full_image_name = '%s/%s' % (path, image_name)
        process_image(full_image_name, window_size, stride, dir_name, txt_fname, label)


def _half_image(image_name, image_num_1, image_num_2):
    dir_name = '/'.join(image_name.split('/')[:-1])

    image = skimage.io.imread(image_name)
    half_w = int(image.shape[1] / 2)
    img_1 = image[:, :half_w, :]
    img_1_name = '%s/archive_2_%d.png' % (dir_name, image_num_1)
    skimage.io.imsave(img_1_name, img_1)

    img_2 = image[:, half_w:, :]
    img_2_name = '%s/archive_2_%d.png' % (dir_name, image_num_2)
    skimage.io.imsave(img_2_name, img_2)


if __name__ == '__main__':
    window_size = 152
    stride = 50
    data_dir = '/home/andrew/Projects/al-maqrizi/data/sw_patches'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    data_fname = '/home/andrew/Projects/al-maqrizi/data/sw_patches/train.txt'

    path = '/home/andrew/Projects/al-maqrizi/data/al-maqrizi/Archive_2/pages'
    label = 1
    process_images_path(path, window_size, stride, data_dir, data_fname, label)

    path = '/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/1/text'
    label = 0
    process_images_path(path, window_size, stride, data_dir, data_fname, label)

    path = '/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/2/text'
    label = 0
    process_images_path(path, window_size, stride, data_dir, data_fname, label)

    path = '/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/3/text'
    label = 0
    process_images_path(path, window_size, stride, data_dir, data_fname, label)
