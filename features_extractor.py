import caffe
import numpy as np
import sliding_window_patches as sw
from skimage import img_as_ubyte
import os


class FeaturesNet(caffe.Net):
    def __init__(self, model_file, pretrained_file, mean_file=None, channel_swap=[2, 1, 0]):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer({in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2, 0, 1))
        if mean_file is not None:
            blob = caffe.proto.caffe_pb2.BlobProto()
            data = open(mean_file, 'rb').read()
            blob.ParseFromString(data)
            arr = np.array(caffe.io.blobproto_to_array(blob))
            out = arr[0]
            self.transformer.set_mean('data', out)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

    def get_features(self, images):
        in_ = self.inputs[0]
        caffe_in = np.zeros((len(images), images[0].shape[2]) + self.blobs[in_].data.shape[2:], dtype=np.float32)
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

    return crops, coords


def process_manuscript(manuscript_path, feat_net, w_size, s):
    features_dir = os.path.join(manuscript_path, 'features')
    if not os.path.exists(features_dir):
        os.mkdir(features_dir)

    image_path = os.path.join(manuscript_path, 'text')
    for fname in os.listdir(image_path):
        print 'Generate features for %s' % fname
        image_name = os.path.join(image_path, fname)
        patches, patches_coord = process_image_sw(image_name, window_size, stride)
        features = feat_net.get_features(map(img_as_ubyte, patches))

        features_fname = os.path.join(features_dir, '%s.txt' % fname.split('.')[0])
        with open(features_fname, 'w') as f:
            for feat in features:
                for i in range(len(feat)):
                    if i != len(feat) - 1:
                        f.write('%s ' % feat[i])
                    else:
                        f.write('%s\n' % feat[i])


if __name__ == '__main__':

    # reload(sw)

    caffe.set_mode_gpu()

    train_images_paths = ['/home/andrew/Projects/al-maqrizi/data/al-maqrizi/Archive_2',
                          '/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/2',
                          '/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/3',
                          '/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/4',
                          '/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/6',
                          '/home/andrew/Projects/al-maqrizi/data/not_al-maqrizi/5',
                          ]

    window_size = 80
    stride = 20

    model = '/home/andrew/Projects/al-maqrizi/nets/sw/deploy_features.prototxt'
    pretrained = '/home/andrew/Projects/al-maqrizi/nets/sw/snapshot_iter_7170.caffemodel'
    mean_fname = '/home/andrew/Projects/al-maqrizi/nets/sw/mean.binaryproto'

    net = FeaturesNet(model, pretrained, mean_file=mean_fname)

    for path in train_images_paths:
        process_manuscript(path, net, window_size, stride)
