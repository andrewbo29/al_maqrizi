import caffe
import numpy as np
import sliding_window_patches as sw
from skimage import img_as_ubyte
import visualize_classified_patches as viz
import matplotlib


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

    def get_features(self, image):
        in_ = self.inputs[0]
        caffe_in = np.zeros((1, image.shape[2]) + self.blobs[in_].data.shape[2:], dtype=np.float32)
        caffe_in[0] = self.transformer.preprocess(in_, image)
        out = self.forward_all(**{in_: caffe_in})
        # predictions = out[self.outputs[0]].squeeze()
        predictions = out[self.outputs[0]]

        return predictions[0].flatten()


def convert_mean(mean_fname, npy_fname):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(mean_fname, 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    out = arr[0]
    np.save(npy_fname, out)


def process_image_sw(image_fname, w_size, s):
    image = sw.normalize_image(image_fname)
    bin_map = sw.get_binary_map(image)
    crops = sw.get_sliding_window_patches(image, bin_map, w_size, s)
    coords = sw.get_sliding_window_patches_coord(image, bin_map, w_size, s)

    for cr, co in zip(crops, coords):
        yield cr, co


# reload(sw)
reload(viz)

MODEL_FILE = '/home/boyarov/Documents/arab/deploy.prototxt'
PRETRAINED = '/home/boyarov/Documents/arab/snapshot_iter_7170.caffemodel'
MEAN = '/home/boyarov/Documents/arab/mean.binaryproto'
MEAN_NPY = '/home/boyarov/Documents/arab/mean.npy'

# convert_mean(MEAN, MEAN_NPY)

caffe.set_mode_gpu()
net = FeaturesNet(MODEL_FILE, PRETRAINED, MEAN)

window_size = 80
stride = 20

IMAGE_FILE = '/home/boyarov/Documents/arab/data/al-maqrizi/Archive_1/Ms-orient-A-01771_014.jpg'
# IMAGE_FILE = '/home/boyarov/Documents/arab/data/not_al-maqrizi/1/Pic-1.png'
# IMAGE_FILE = '/home/boyarov/Documents/arab/data/not_al-maqrizi/4/Pic-1.png'

prob = []
bbox_list = []
for patch, coord in process_image_sw(IMAGE_FILE, window_size, stride):
    output = net.get_features(img_as_ubyte(patch))
    prob.append(output[1])
    bbox_list.append((coord, output[1]))

matplotlib.rcParams['figure.figsize'] = (10.0, 18.0)
viz.plot_image_estimated_al_maqrizi_probability2(sw.normalize_image(IMAGE_FILE), bbox_list)