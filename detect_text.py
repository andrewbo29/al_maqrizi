from connected_components_patches import *
import os
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
from spsa_clustering import *


def plot_component(img_fname, bbox, color):
    img = plt.imread(img_fname)
    plt.imshow(img)
    currentAxis = plt.gca()
    coords = (bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0]
    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=3))


def get_component_center(bbox):
    return [bbox[1] + float(bbox[3] - bbox[1]) / 2, bbox[0] + float(bbox[2] - bbox[0]) / 2]


def get_intensity(img):
    return np.mean(img)


# if __name__ == '__main__':
image_file = '/home/andrew/Projects/al-maqrizi/data/al-maqrizi/Archive_1/norm/Ms-orient-A-01771_021.jpg'

regions_filter = RegionsFilter()
region_resizer = MultiplyLowerThanMedianRegionResizer(mul=1.2)

crops_with_bboxes = extract_patches(
    input_file=image_file,
    output_images_height=28,
    output_images_width=28,
    regions_filter=regions_filter,
    region_resizer=region_resizer,
    output_binary=True,
    return_bbox=True
)

crops_images, crops_bboxes = zip(*crops_with_bboxes)

# for crop in np.array(crops_bboxes):
#     plot_component(image_file, crop, 'r')
# plt.show()

centers = np.array(map(get_component_center, crops_bboxes))
#
# intensity = np.array(map(np.mean, crops_images))

# print centers.shape

# plot_component(image_file, crops_bboxes[100])

# kmeans = KMeans(n_clusters=2)
# res = kmeans.fit_predict(centers)
# res = kmeans.fit_predict(intensity[:,np.newaxis])

# clustering = ClusteringSPSA(n_clusters=2)
# for data_point in intensity[:,np.newaxis]:
#     clustering.fit(data_point)
# clustering.clusters_fill(intensity[:,np.newaxis])
# res = clustering.labels_

# res = SpectralClustering(n_clusters=2, affinity='rbf', gamma=0.5).fit_predict(centers)
#
# for crop in np.array(crops_bboxes)[res == 0]:
#     plot_component(image_file, crop, 'r')
# for crop in np.array(crops_bboxes)[res == 1]:
#     plot_component(image_file, crop, 'b')
# plt.show()
