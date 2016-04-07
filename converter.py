import skimage.io
import os
import shutil

image_path = '/home/andrew/Projects/al-maqrizi/data/al-maqrizi_plagiarism'
converted_path = '/home/andrew/Projects/al-maqrizi/data/al-maqrizi_plagiarism/convert'

for fname in os.listdir(image_path):
    if os.path.isfile(os.path.join(image_path, fname)):
        if fname.endswith('.tiff'):
            img = skimage.io.imread(os.path.join(image_path, fname))
            img_name = fname.split('.')[0]
            skimage.io.imsave(os.path.join(converted_path, '%s.jpg' % img_name), img)
        else:
            shutil.copy(os.path.join(image_path, fname), os.path.join(converted_path, fname))
