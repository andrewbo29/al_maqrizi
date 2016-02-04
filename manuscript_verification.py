import numpy as np
import matplotlib.pyplot as plt
import caffe
import sliding_window_patches as sw


class VerificationNet(caffe.Net):
    def __init__(self, model_file, pretrained_file, channel_swap=[2, 1, 0]):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer({in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2, 0, 1))
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

    def get_features(self, windows):
        in_ = self.inputs[0]
        caffe_in = np.zeros((len(windows), windows[0].shape[2]) + self.blobs[in_].data.shape[2:], dtype=np.float32)
        for ix, window_in in enumerate(windows):
            caffe_in[ix] = self.transformer.preprocess(in_, window_in)
        out = self.forward_all(**{in_: caffe_in})
        # predictions = out[self.outputs[0]].squeeze()
        predictions = out[self.outputs[0]]

        features = []
        ix = 0
        for ix in range(len(windows)):
            features.append(predictions[ix].flatten())
            ix += 1
        return np.array(features)


# if __name__ == '__main__':
input_image_fname = '/home/andrew/Projects/al-maqrizi/data/hitat/text/UM605 pgs. 002-003.JPG_1.png'

image = sw.normalize_image(input_image_fname)

window_size = 80
stride = 20
bin_map = sw.get_binary_map(image)
crops = sw.get_sliding_window_patches(image, bin_map, window_size, stride)

model = '/home/andrew/digits/digits/jobs/20160203-211622-9ea3/deploy.prototxt'
pretrained = '/home/andrew/digits/digits/jobs/20160203-211622-9ea3/snapshot_iter_7170.caffemodel'
net = VerificationNet(model, pretrained)

caffe.set_mode_gpu()
res = net.get_features(crops)
