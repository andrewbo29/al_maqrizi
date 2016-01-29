#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import math
import time
import numpy as np
import scipy as sp

import skimage
import skimage.io
import skimage.morphology
import skimage.filters
import skimage.transform
import skimage.draw
import skimage.measure

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import argparse


def otsu(img, radius=20):
    selem = skimage.morphology.disk(radius)
    threshold_global_otsu = skimage.filters.threshold_otsu(img)
    global_otsu = img >= threshold_global_otsu
    return global_otsu * 1


class ArgParseMixin(object):
    def add_args(self, parser):
        """
        Add arguments specifications to argument parser

        :type parser: argparse.ArgumentParser
        """
        raise NotImplementedError()

    def parse_args(self, args):
        """
        Read values of arguments specified in `add_args` method

        :type args: output of argparse.ArgumentParser.parse_args method 
        """
        raise NotImplementedError()


class RegionsFilter(ArgParseMixin):
    def __init__(
            self,
            max_major_minor_axis_length_ratio=10.,
            min_minor_axis_length=8,
            min_major_axis_length=15,
            max_major_axis_length=160,
            dbscan_eps=0.3,
            dbscan_min_samples=10
    ):
        self.max_major_minor_axis_length_ratio = max_major_minor_axis_length_ratio
        self.min_minor_axis_length = min_minor_axis_length
        self.min_major_axis_length = min_major_axis_length
        self.max_major_axis_length = max_major_axis_length
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

    def add_args(self, parser):
        parser.add_argument('--dbscan_eps', type=float,
                            help="eps parameter for DBSCAN algorithm used to detect outlier regions", default=0.3)
        parser.add_argument('--dbscan_min_samples', type=int,
                            help="min_samples parameter for DBSCAN algorithm used to detect outlier regions",
                            default=10)
        parser.add_argument('--max_major_axis_length', type=int,
                            help="filter regions with major_axis_length greater then this value", default=160)
        parser.add_argument('--min_major_axis_length', type=int,
                            help="filter regions with major_axis_length lower then this value", default=15)
        parser.add_argument('--min_minor_axis_length', type=int,
                            help="filter regions with minor_axis_length lower then this value", default=7)
        parser.add_argument('--max_major_minor_axis_length_ratio', type=float,
                            help="filter regions with major_axis_length/minor_axis_length ration greater then this value",
                            default=10.)

    def parse_args(self, args):
        self.__init__(
                dbscan_eps=args.dbscan_eps,
                dbscan_min_samples=args.dbscan_min_samples,
                max_major_axis_length=args.max_major_axis_length,
                min_major_axis_length=args.min_major_axis_length,
                min_minor_axis_length=args.min_minor_axis_length,
                max_major_minor_axis_length_ratio=args.max_major_minor_axis_length_ratio
        )

    def _accept_region_prop(self, rp):
        if rp.minor_axis_length == 0:
            return False
        if rp.major_axis_length / rp.minor_axis_length > self.max_major_minor_axis_length_ratio:
            return False
        if rp.minor_axis_length < self.min_minor_axis_length:
            return False
        if rp.major_axis_length < self.min_major_axis_length:
            return False
        if rp.major_axis_length > self.max_major_axis_length:
            return False
        return True

    def _filter_regions_by_properties(self, regions):
        return filter(self._accept_region_prop, regions)

    def _filter_regions_by_position(self, regions):
        X = np.array([rp.centroid for rp in regions])
        X = StandardScaler().fit_transform(X)
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(X)
        labels = db.labels_
        return [r for r, l in zip(regions, labels) if l >= 0]

    def filter(self, regions):
        regions_filtered = self._filter_regions_by_properties(regions)
        regions_filtered = self._filter_regions_by_position(regions_filtered)
        return regions_filtered


class AbstractRegionResizer(object):
    def resize(self, image, bbox):
        """
        Copy region of image

        :type image: numpy.ndarray (2-dimensional or 3-dimensional of shape (n,m,3))
        :type region: skimage.measure._regionprops._RegionProperties

        :rtype: tuple of four ints (minr, minc, maxr, maxc) 
        """
        raise NotImplementedError()

    def fit(self, regions):
        pass


class MultiplyRegionResizer(ArgParseMixin, AbstractRegionResizer):
    def __init__(self, mul=1):
        self.mul = mul

    def add_args(self, parser):
        parser.add_argument('--bbox_mul', type=float, help="enlarge output patches bounding box by given factor",
                            default=1.2)

    def parse_args(self, args):
        self.mul = args.bbox_mul

    def resize(self, bbox):
        minr, minc, maxr, maxc = enlarge_bbox(bbox, self.mul)
        return minr, minc, maxr, maxc


class LowerThanMedianRegionResizer(ArgParseMixin, AbstractRegionResizer):
    def __init__(self):
        self.height_median = None
        self.width_median = None

    def add_args(self, parser):
        pass

    def parse_args(self, args):
        pass

    def fit(self, regions):
        self.height_median = np.median(map(get_region_height, regions))
        self.width_median = np.median(map(get_region_width, regions))

    def resize(self, image, bbox):
        minr, minc, maxr, maxc = bbox
        if maxr - minr < self.width_median:
            r = (maxr + minr) / 2.
            maxr = int(r + self.width_median / 2.)
            minr = int(r - self.width_median / 2.)
        if maxc - minc < self.height_median:
            c = (maxc + minc) / 2.
            maxc = int(c + self.height_median / 2.)
            minc = int(c - self.height_median / 2.)
        return minr, minc, maxr, maxc


class MultiplyLowerThanMedianRegionResizer(ArgParseMixin, AbstractRegionResizer):
    def __init__(self, mul=1):
        self.mul = mul
        self.height_median = None
        self.width_median = None

    def add_args(self, parser):
        parser.add_argument('--bbox_mul', type=float, help="enlarge output patches bounding box by given factor",
                            default=1.2)

    def parse_args(self, args):
        self.mul = args.bbox_mul

    def fit(self, regions):
        self.height_median = np.median(map(get_region_height, regions))
        self.width_median = np.median(map(get_region_width, regions))

    def resize(self, bbox):
        minr, minc, maxr, maxc = bbox
        if maxr - minr < self.width_median:
            r = (maxr + minr) / 2.
            maxr = int(r + self.width_median / 2.)
            minr = int(r - self.width_median / 2.)
        if maxc - minc < self.height_median:
            c = (maxc + minc) / 2.
            maxc = int(c + self.height_median / 2.)
            minc = int(c - self.height_median / 2.)
        minr, minc, maxr, maxc = enlarge_bbox((minr, minc, maxr, maxc), self.mul)
        return minr, minc, maxr, maxc


def enlarge_bbox(bbox, mul):
    minr, minc, maxr, maxc = bbox
    c_w = int((maxc - minc) * mul / 2.)
    r_w = int((maxr - minr) * mul / 2.)
    r = (minr + maxr) / 2.
    c = (minc + maxc) / 2.
    return minr, minc, maxr, maxc


def extract_regions(image):
    label_img = skimage.measure.label(image)
    regions = skimage.measure.regionprops(label_img)
    return regions


def get_region_height(r):
    return r.bbox[3] - r.bbox[1]


def get_region_width(r):
    return r.bbox[2] - r.bbox[0]


def coalesce(*args):
    for v in args:
        if v is not None:
            return v
    return None


def resize2d(im, shape):
    if len(im.shape) == 3:
        return skimage.transform.resize(im, (shape[0], shape[1], im.shape[2]))
    if len(im.shape) == 2:
        return skimage.transform.resize(im, (shape[0], shape[1]))


def extract_patches(
        input_file,
        output_images_height,
        output_images_width,
        regions_filter,
        region_resizer,
        output_binary=False
):
    # reading input image
    image_color = skimage.io.imread(input_file, as_grey=False)
    # otsu binarization
    image = 1 - otsu(skimage.io.imread(input_file, as_grey=True))

    # regions extraction
    regions = extract_regions(image)
    print 'extracted regions: {}'.format(len(regions))
    # regions filtering
    regions_filtered = regions_filter.filter(regions)
    print 'filtered regions: {}'.format(len(regions_filtered))

    region_resizer.fit(regions_filtered)

    # selecting image to output
    if output_binary:
        image_for_output = image * 1.
    else:
        image_for_output = image_color * 1.
    if image_for_output.max() > 1.:
        image_for_output /= 255.

    for i, r in enumerate(regions_filtered):
        # resizing region bounding box
        minr, minc, maxr, maxc = region_resizer.resize(r.bbox)

        # fixing bounding box
        minr = max(0, minr)
        minc = max(0, minc)
        maxr = min(image_for_output.shape[0], maxr)
        maxc = min(image_for_output.shape[1], maxc)

        # copying patch from output image
        if len(image_for_output.shape) == 3:
            patch = image_for_output[minr:maxr, minc:maxc, :]
        else:
            patch = image_for_output[minr:maxr, minc:maxc]

        # resizing patch to specified shape
        patch_resized = resize2d(patch, (output_images_height, output_images_width))
        yield patch_resized


def save_images_in_dir(output_dir, images, output_format, txt_fname, label, prefix=''):
    with open(txt_fname, 'a') as txt_file:
        for i, image in enumerate(images):
            image_fname = os.path.join(output_dir, '{0}{1}.{2}'.format(prefix, i, output_format))
            skimage.io.imsave(image_fname, image)
            txt_file.write('%s %d\n' % (image_fname, label))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Extract patches from given file to given dir basing on otsu-binarization and connected componetnts detection')

    parser.add_argument('-i', '--input_file', type=str, help="input file path", required=True)
    parser.add_argument('-o', '--output_dir', type=str, help="output directory path", required=True)
    parser.add_argument('-of', '--output_format', type=str, help="output files format", default='png')
    parser.add_argument('-ob', '--output_binary', help="write binarized images to output", default=False,
                        action='store_true')
    parser.add_argument('-oiw', '--output_images_width', type=int, help="output images width", required=False,
                        default=None)
    parser.add_argument('-oih', '--output_images_height', type=int, help="output images height", required=False,
                        default=None)

    # specifying objects for further use
    regions_filter = RegionsFilter()
    region_resizer = MultiplyLowerThanMedianRegionResizer()

    # add objects arguments
    regions_filter.add_args(parser)
    region_resizer.add_args(parser)

    # parse command-line arguments
    args = parser.parse_args()
    regions_filter.parse_args(args)
    region_resizer.parse_args(args)

    time_start = time.clock()

    patches = extract_patches(
            args.input_file,
            args.output_images_height,
            args.output_images_width,
            regions_filter,
            region_resizer,
            args.output_binary,
    )

    save_images_in_dir(
            args.output_dir,
            patches,
            args.output_format
    )

    time_end = time.clock()

    print 'Files {0}--{1}.{2} saved to directory {3} in {4} seconds'.format(
            0, i - 1, output_format, output_dir, time_end - time_start
    )
