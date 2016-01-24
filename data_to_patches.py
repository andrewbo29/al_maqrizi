#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from connected_components_patches import *


POSITIVE_IMAGES_DIR_LIST = [
	'data/al-maqrizi/Archive_1',
	'data/al-maqrizi/Archive_2'
]

NEGATIVE_IMAGES_DIR_LIST = [
	'data/not_al-maqrizi/{}'.format(i) for i in range(1, 9)
]


OUTPUT_DIR = 'patches'

SAMPLES_TYPES = ['train', 'val', 'test']

CLASS_IMAGES_DIR_LIST = [(1, d) for d in POSITIVE_IMAGES_DIR_LIST] + [(0, d) for d in NEGATIVE_IMAGES_DIR_LIST]

for s in SAMPLES_TYPES:
	for c in [0, 1]:
		try:
			os.makedirs(os.path.join(OUTPUT_DIR, s, str(c)))
		except OSError:
			pass


regions_filter = RegionsFilter()
region_resizer = MultiplyLowerThanMedianRegionResizer(mul=1.2)


for class_label, images_dir in CLASS_IMAGES_DIR_LIST:
	dir_name = images_dir.split(os.sep)[-1]

	for sample_type in SAMPLES_TYPES:
		output_sample_dir = os.path.join(OUTPUT_DIR, sample_type, str(class_label))
		images_sample_dir = os.path.join(images_dir, sample_type)
		images_files = [os.listdir(images_sample_dir)[0]]

		for image_file in images_files:
			time_start = time.clock()
			patches = extract_patches(
		        input_file=os.path.join(images_sample_dir, image_file),
		        output_images_height=20,
		        output_images_width=20,
		        regions_filter=regions_filter,
		        region_resizer=region_resizer,
		        output_binary=False
		        )

			save_images_in_dir(
				output_sample_dir,
				patches,
				'png',
				prefix='__'.join([dir_name, image_file.split('.')[0]]) + '__'
				)
			time_end = time.clock()
			print 'File {0} done in {1} seconds'.format(image_file, time_end - time_start)

