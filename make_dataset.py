import pandas as pd
import os
import argparse

opt = argparse.ArgumentParser()
opt.add_argument('--filepath', type=str, help='input your own flower path', default='/home/monkey/Project/RcnnApp/17flowers/jpg')


filepath = opt.filepath
filenames = os.listdir(filepath)
print(filenames)

with open('dataset.csv', 'a') as f:
    for name in filenames:
        sub_filepath = os.path.join(filepath, name)
        images = os.listdir(sub_filepath)
        for image in images:
            f.write('{},{}'.format(os.path.join(sub_filepath, image), int(name)))
            f.write('\n')

