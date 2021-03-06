import caffe
import numpy as np
import sliding_window_patches as sw
import connected_components_patches as ccp
from skimage import img_as_ubyte
import visualize_classified_patches as viz
import matplotlib
import os
from skimage.color import rgb2gray


class VerificationNet(caffe.Net):
    def __init__(self, model_file, pretrained_file, mean_file=None, channel_swap=[2, 1, 0]):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer({in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2, 0, 1))
        if mean_file is not None:
            blob = caffe.proto.caffe_pb2.BlobProto()
            # blob = caffe.io.caffe_pb2.BlobProto()
            data = open(mean_file, 'rb').read()
            blob.ParseFromString(data)
            arr = np.array(caffe.io.blobproto_to_array(blob))
            out = arr[0]
            self.transformer.set_mean('data', out)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

    def predict(self, images):
        in_ = self.inputs[0]
        caffe_in = np.zeros((len(images), images[0].shape[2]) + self.blobs[in_].data.shape[2:], dtype=np.float32)
        # caffe_in = np.zeros((len(images), 1) + self.blobs[in_].data.shape[2:], dtype=np.float32)
        for ix, img in enumerate(images):
            caffe_in[ix] = self.transformer.preprocess(in_, img)
        out = self.forward_all(**{in_: caffe_in})
        predictions = out[self.outputs[0]]

        return predictions


def process_image_sw(image_fname, w_size, s):
    image = sw.normalize_image(image_fname)
    bin_map = sw.get_binary_map(image)
    crops = sw.get_sliding_window_patches(image, bin_map, w_size, s)
    coords = sw.get_sliding_window_patches_coord(image, bin_map, w_size, s)

    # gray_crops = [rgb2gray(crop) for crop in crops]

    return crops, coords
    # return gray_crops, coords


def process_image_components(image_fname, size):
    regions_filter = ccp.RegionsFilter()
    region_resizer = ccp.MultiplyLowerThanMedianRegionResizer(mul=1.2)

    crops_with_bboxes = ccp.extract_patches(
            input_file=image_fname,
            output_images_height=size,
            output_images_width=size,
            regions_filter=regions_filter,
            region_resizer=region_resizer,
            output_binary=True,
            return_bbox=True
    )

    crops_images, crops_bboxes = zip(*crops_with_bboxes)
    new_crops_images = []
    for img in crops_images:
        if img.ndim == 2:
            # img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
            img = img[:, :, np.newaxis]
            # img = img[np.newaxis, :, :]
            new_crops_images.append(img)

    return new_crops_images, crops_bboxes


def convert_mean(mean_fname, npy_fname):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(mean_fname, 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    out = arr[0]
    np.save(npy_fname, out)


if __name__ == '__main__':

    # reload(sw)
    # reload(viz)

    caffe.set_mode_gpu()

    patches_type = 'sw'
    # patches_type = 'components'

    # image_path = '/home/andrew/Projects/al-maqrizi/data/hitat/text'
    # image_path = '/home/andrew/Projects/al-maqrizi/data/al-maqrizi/Archive_1/text'
    # image_path = '/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/8/text'
    # image_path = '/home/andrew/Projects/al-maqrizi/data/al-maqrizi_plagiarism/convert'
    image_path = '/home/andrew/Projects/al-maqrizi/data/samples/converted'

    if patches_type == 'sw':
        window_size = 80
        stride = 20

        model = '/home/andrew/Projects/al-maqrizi/nets/sw/deploy.prototxt'
        pretrained = '/home/andrew/Projects/al-maqrizi/nets/sw/snapshot_iter_7170.caffemodel'
        mean_fname = '/home/andrew/Projects/al-maqrizi/nets/sw/mean.binaryproto'

        # model = '/home/andrew/digits/digits/jobs/20160504-231044-9215/deploy.prototxt'
        # pretrained = '/home/andrew/digits/digits/jobs/20160504-231044-9215/snapshot_iter_5610.caffemodel'
        # mean_fname = '/home/andrew/digits/digits/jobs/20160504-220619-1eef/mean.binaryproto'

        net = VerificationNet(model, pretrained, mean_file=mean_fname)

        doc_probs = []
        for fname in os.listdir(image_path):
            print 'Process %s' % fname

            image_name = os.path.join(image_path, fname)

            patches, patches_coord = process_image_sw(image_name, window_size, stride)

            output = net.predict(map(img_as_ubyte, patches))
            bbox_list = [(patches_coord[i], output[i][1]) for i in range(len(output))]

            doc_probs.append(np.mean([output[i][1] for i in range(len(output))]))

            matplotlib.rcParams['figure.figsize'] = (10.0, 18.0)
            viz.plot_image_estimated_al_maqrizi_probability2(sw.normalize_image(image_name), bbox_list)

        print 'Al-Maqrizi document probability: %f' % np.mean(doc_probs)

    elif patches_type == 'components':
        patch_size = 28

        model = '/home/andrew/Projects/al-maqrizi/nets/components_bin/deploy.prototxt'
        pretrained = '/home/andrew/Projects/al-maqrizi/nets/components_bin/snapshot_iter_70100.caffemodel'
        mean_file = '/home/andrew/Projects/al-maqrizi/nets/components_bin/mean.npy'

        net = caffe.Classifier(model, pretrained, image_dims=(patch_size, patch_size),
                               mean=np.load(mean_file).mean(1).mean(1))

        doc_probs = []
        for fname in os.listdir(image_path):
            print 'Process %s' % fname

            image_name = os.path.join(image_path, fname)

            patches, patches_coord = process_image_components(image_name, size=patch_size)

            output = net.predict(map(img_as_ubyte, patches))
            bbox_list = [(patches_coord[i], output[i][1]) for i in range(len(output))]

            doc_probs.append(np.mean([output[i][1] for i in range(len(output))]))

            matplotlib.rcParams['figure.figsize'] = (10.0, 18.0)
            viz.plot_image_estimated_al_maqrizi_probability2(sw.normalize_image(image_name), bbox_list)

        print 'Al-Maqrizi document probability: %f' % np.mean(doc_probs)
