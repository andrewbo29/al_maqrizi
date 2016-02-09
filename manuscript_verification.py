import caffe
import numpy as np
import sliding_window_patches as sw
from skimage import img_as_ubyte
import visualize_classified_patches as viz
import matplotlib


class VerificationNet(caffe.Net):
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


def process_image_sw(image_fname, w_size, s):
    image = sw.normalize_image(image_fname)
    bin_map = sw.get_binary_map(image)
    crops = sw.get_sliding_window_patches(image, bin_map, w_size, s)
    coords = sw.get_sliding_window_patches_coord(image, bin_map, w_size, s)

    for cr, co in zip(crops, coords):
        yield cr, co


if __name__ == '__main__':

    # reload(sw)
    reload(viz)

    window_size = 80
    stride = 20

    model = '/home/andrew/digits/digits/jobs/20160203-211622-9ea3/deploy.prototxt'
    pretrained = '/home/andrew/digits/digits/jobs/20160203-211622-9ea3/snapshot_iter_7170.caffemodel'
    mean_fname = '/home/andrew/digits/digits/jobs/20160203-204818-3c44/mean.binaryproto'

    caffe.set_mode_gpu()
    net = VerificationNet(model, pretrained, mean_file=mean_fname)

    # image_name = '/home/boyarov/Documents/arab/data/al-maqrizi/Archive_1/Ms-orient-A-01771_014.jpg'
    # image_name = '/home/andrew/Projects/al-maqrizi/data/al-maqrizi/Archive_1/text/Ms-orient-A-01771_019.jpg'
    image_name = '/home/andrew/Projects/al-maqrizi/data/hitat/text/UM605 pgs. 006-007.JPG_1.png'

    prob = []
    bbox_list = []
    for patch, coord in process_image_sw(image_name, window_size, stride):
        output = net.get_features(img_as_ubyte(patch))
        prob.append(output[1])
        bbox_list.append((coord, output[1]))

    matplotlib.rcParams['figure.figsize'] = (10.0, 18.0)
    viz.plot_image_estimated_al_maqrizi_probability2(sw.normalize_image(image_name), bbox_list)
