import os
from PIL import Image


def convert(path, tiff_fname):
    if not os.path.isfile(tiff_fname):
        return
    im = Image.open(tiff_fname)
    print "Generating jpeg for %s" % tiff_fname
    im.thumbnail(im.size)
    name = tiff_fname.split(os.path.sep)[-1]
    outfile = os.path.join(path, 'converted', '%s.jpg' % name.split('.')[0])
    im.save(outfile, "JPEG", quality=100)


root = '/home/andrew/Projects/al-maqrizi/data/samples'
for fname in os.listdir(root):
    convert(root, os.path.join(root, fname))