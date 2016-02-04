import numpy as np
import skimage.io
import skimage.filters
import skimage.transform
import skimage.color
import os

RESIZE_HEIGHT = 700
RESIZE_WEIGHT = 500

NORMALIZE_THRESH_DOWN = 0.1
NORMALIZE_THRESH_UP = 0.97

VALID_PATCH_THRESH = 0.9

PATCH_THRESH_DOWN = 0
PATCH_THRESH_UP = 0.73


def get_binary_map(image):
    grey_image = skimage.color.rgb2grey(image)

    global_thresh = skimage.filters.threshold_otsu(grey_image)
    binary_global = grey_image > global_thresh

    return np.where(binary_global, grey_image, 0.)


def is_valid_patch(patch):
    height = patch.shape[0]
    weight = patch.shape[1]

    for i in range(height):
        res = float((patch[i, :] == 0).sum()) / weight
        if res >= VALID_PATCH_THRESH:
            return False

    for j in range(weight):
        res = float((patch[:, j] == 0).sum()) / height
        if res >= VALID_PATCH_THRESH:
            return False

    return True


def get_sliding_window_patches(image, binary_map, window_size=152, stride=10):
    x1 = 0
    y1 = 0
    x2 = window_size
    y2 = window_size

    patches = []
    while x2 < binary_map.shape[0]:
        binary_patch = binary_map[x1:x2, y1:y2]
        if PATCH_THRESH_DOWN < np.mean(binary_patch) < PATCH_THRESH_UP:
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


def _show_rand_patch(patches_list):
    ind = np.random.choice(len(patches_list))
    skimage.io.imshow(patches_list[ind])
    skimage.io.show()


def _show_patches(patches_list):
    for ind in range(len(patches_list)):
        skimage.io.imshow(patches_list[ind])
        skimage.io.show()


def _show_patch(patches_list, patch_ind):
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

    image = normalize_image(image_name)

    bin_map = get_binary_map(image)
    crops = get_sliding_window_patches(image, bin_map, window_size, stride)
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


def normalize_image(image_name):
    origin_image = skimage.io.imread(image_name)

    bin_map = get_binary_map(origin_image)

    height = bin_map.shape[0]
    weight = bin_map.shape[1]

    x_up = 0
    x_down = height
    y_left = 0
    y_right = weight

    for i in range(height):
        t = np.median(bin_map[i, :]) - 0.02
        if t < NORMALIZE_THRESH_DOWN or t > NORMALIZE_THRESH_UP:
            x_up = i
        else:
            break

    for i in range(height - 1, 0, -1):
        t = np.median(bin_map[i, :]) + 0.02
        if t < NORMALIZE_THRESH_DOWN or t > NORMALIZE_THRESH_UP:
            x_down = i
        else:
            break

    for j in range(weight):
        t = np.median(bin_map[:, j])
        if t < NORMALIZE_THRESH_DOWN or t > NORMALIZE_THRESH_UP:
            y_left = j
        else:
            break

    for j in range(weight - 1, 0, -1):
        t = np.median(bin_map[:, j]) - 0.01
        if t < NORMALIZE_THRESH_DOWN or t > NORMALIZE_THRESH_UP:
            y_right = j
        else:
            break

    new_image = origin_image[x_up:x_down, y_left:y_right]

    return skimage.transform.resize(new_image, (RESIZE_HEIGHT, RESIZE_WEIGHT))


def _generate_normalize_images(path):
    img_path = os.path.join(path, 'text')
    for fname in os.listdir(img_path):
        image_name = os.path.join(img_path, fname)
        norm_dir = os.path.join(path, 'norm')
        if not os.path.exists(norm_dir):
            os.mkdir(norm_dir)
        norm_image_name = os.path.join(norm_dir, fname)
        skimage.io.imsave(norm_image_name, normalize_image(image_name))


def _generate_normalize_double_images(image_name):
    origin_image = skimage.io.imread(image_name)

    bin_map = get_binary_map(origin_image)

    height = bin_map.shape[0]
    weight = bin_map.shape[1]

    x_up = 0
    x_down = height
    y_left = 0
    y_right = weight

    for i in range(height):
        t = np.median(bin_map[i, :]) - 0.02
        if t < NORMALIZE_THRESH_DOWN or t > NORMALIZE_THRESH_UP:
            x_up = i
        else:
            break

    for i in range(height - 1, 0, -1):
        t = np.median(bin_map[i, :]) + 0.02
        if t < NORMALIZE_THRESH_DOWN or t > NORMALIZE_THRESH_UP:
            x_down = i
        else:
            break

    for j in range(weight):
        t = np.median(bin_map[:, j])
        if t < NORMALIZE_THRESH_DOWN or t > NORMALIZE_THRESH_UP:
            y_left = j
        else:
            break

    for j in range(weight - 1, 0, -1):
        t = np.median(bin_map[:, j]) - 0.01
        if t < NORMALIZE_THRESH_DOWN or t > NORMALIZE_THRESH_UP:
            y_right = j
        else:
            break

    new_image = origin_image[x_up:x_down, y_left:y_right]

    dir_name = '/'.join(image_name.split('/')[:-1])
    image_fname = image_name.split('/')[-1]

    half_w = int(new_image.shape[1] / 2)
    img_1 = new_image[:, :half_w + 100, :]
    img_1_name = '%s/text/%s_1.png' % (dir_name, image_fname)
    skimage.io.imsave(img_1_name, img_1)

    img_2 = new_image[:, half_w:, :]
    img_2_name = '%s/text/%s_2.png' % (dir_name, image_fname)
    skimage.io.imsave(img_2_name, img_2)


if __name__ == '__main__':
    train_images_paths = [('/home/andrew/Projects/al-maqrizi/data/al-maqrizi/Archive_2/text', 1),
                          ('/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/2/text', 0),
                          ('/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/3/text', 0),
                          ('/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/4/text', 0),
                          ('/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/6/text', 0),
                          ('/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/5/text', 0),
                          ]

    val_images_paths = [('/home/andrew/Projects/al-maqrizi/data/al-maqrizi/Archive_1/text', 1),
                        ('/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/1/text', 0),
                        ('/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/7/text', 0),
                        ('/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/8/text', 0)]

    window_size = 80
    stride = 20
    data_dir = '/home/andrew/Projects/al-maqrizi/data/sw_patches'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    train_fname = '/home/andrew/Projects/al-maqrizi/data/sw_patches/train.txt'
    val_fname = '/home/andrew/Projects/al-maqrizi/data/sw_patches/val.txt'

    for image_path, class_label in train_images_paths:
        process_images_path(image_path, window_size, stride, data_dir, train_fname, class_label)

    for image_path, class_label in val_images_paths:
        process_images_path(image_path, window_size, stride, data_dir, val_fname, class_label)

        # for image_path in images_paths:
        #     _generate_normalize_images(image_path)

        # _generate_normalize_images('/home/andrew/Projects/al-maqrizi/data/al-maqrizi/Archive_1')

        # process_image('/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/5/norm/5_7.png', window_size, stride, data_dir, train_fname, 0)

        # path = '/home/andrew/Projects/al-maqrizi/data/hitat'
        # for image_name in os.listdir(path):
        #     full_image_name = os.path.join(path, image_name)
        #     if os.path.isfile(full_image_name):
        #         _generate_normalize_double_images(full_image_name)
