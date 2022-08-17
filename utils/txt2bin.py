import numpy as np
from glob import glob
import os


for file in glob('dsor_txt/*'):
    labels = np.loadtxt(file)
    filename = os.path.join('dsor_result', file.split('/')[-1].split('.')[0] + '.bin')
    labels.astype('uint32').tofile(filename)
