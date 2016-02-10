from connected_components_patches import *
import os
import skimage
import skimage.color


def fix_if_grayscale(im):
    if len(im.shape) == 3:
        return im
    if len(im.shape) == 2:
        return skimage.color.gray2rgb(im)
    raise BaseException('Unsupported image shape: {}'.format(str(im.shape)))


def process_images_path(path, dir_name, txt_fname, label):
    regions_filter = RegionsFilter()
    region_resizer = MultiplyLowerThanMedianRegionResizer(mul=1.2)

    for image_file in os.listdir(path):
        print image_file
        crops_with_bboxes = extract_patches(
                input_file=os.path.join(path, image_file),
                output_images_height=28,
                output_images_width=28,
                regions_filter=regions_filter,
                region_resizer=region_resizer,
                output_binary=True,
                return_bbox=True
        )
        crops_images, crops_bboxes = zip(*crops_with_bboxes)
        crops_images = map(fix_if_grayscale, crops_images)
        output_sample_dir = os.path.join(dir_name, image_file[:-4])
        if not os.path.exists(output_sample_dir):
            os.mkdir(output_sample_dir)

        save_images_in_dir(
                output_sample_dir,
                crops_images,
                'png',
                txt_fname,
                label
        )


if __name__ == '__main__':
    # root_dir = '.'
    root_dir = '/home/andrew/Projects/al-maqrizi'
    # root_dir = '/home/boyarov/Projects/al-maqrizi'
    in_root_dir = lambda p: os.path.join(root_dir, p)

    data_dir = in_root_dir('data/components_patches_bin')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    data_fname = in_root_dir('data/components_patches_bin/train.txt')

    # MANUSCRIPTS = [(in_root_dir('data/al-maqrizi/Archive_2/norm'), 1),
    #                (in_root_dir('data/not_al-maqrizi/1/norm'), 0),
    #                (in_root_dir('data/not_al-maqrizi/2/norm'), 0),
    #                (in_root_dir('data/not_al-maqrizi/3/norm'), 0),
    #                (in_root_dir('data/not_al-maqrizi/4/norm'), 0),
    #                (in_root_dir('data/not_al-maqrizi/5/norm'), 0),
    #                (in_root_dir('data/not_al-maqrizi/8/norm'), 0)]

    data_fname = in_root_dir('data/components_patches_bin/val.txt')

    MANUSCRIPTS = [(in_root_dir('data/al-maqrizi/Archive_1/norm'), 1),
                   (in_root_dir('data/not_al-maqrizi/6/norm'), 0),
                   (in_root_dir('data/not_al-maqrizi/8/norm'), 0)]

    for manuscript_path, class_label in MANUSCRIPTS:
        process_images_path(manuscript_path, data_dir, data_fname, class_label)
