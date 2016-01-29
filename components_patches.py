from connected_components_patches import *
import os


def process_images_path(path, dir_name, txt_fname, label):
    regions_filter = RegionsFilter()
    region_resizer = MultiplyLowerThanMedianRegionResizer(mul=1.2)

    for image_file in os.listdir(path):
        crops = extract_patches(
                input_file=os.path.join(path, image_file),
                output_images_height=32,
                output_images_width=32,
                regions_filter=regions_filter,
                region_resizer=region_resizer,
                output_binary=False
        )

        output_sample_dir = os.path.join(dir_name, image_file[:-4])
        if not os.path.exists(output_sample_dir):
            os.mkdir(output_sample_dir)

        save_images_in_dir(
                output_sample_dir,
                crops,
                'png',
                txt_fname,
                label
        )


data_dir = '/home/andrew/Projects/al-maqrizi/data/components_patches'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

# data_fname = '/home/andrew/Projects/al-maqrizi/data/components_patches/train.txt'
#
# MANUSCRIPTS = [('/home/andrew/Projects/al-maqrizi/data/al-maqrizi/Archive_2/pages', 1),
#                ('/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/1/text', 0),
#                ('/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/2/text', 0),
#                ('/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/3/text', 0),
#                ('/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/4/text', 0),
#                ('/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/5/text', 0)]

data_fname = '/home/andrew/Projects/al-maqrizi/data/components_patches/val.txt'

MANUSCRIPTS = [('/home/andrew/Projects/al-maqrizi/data/al-maqrizi/Archive_1', 1),
               ('/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/6/text', 0),
               ('/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/7/text', 0),
               ('/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/8/text', 0)]

for manuscript_path, class_label in MANUSCRIPTS:
    process_images_path(manuscript_path, data_dir, data_fname, class_label)
