from __future__ import division, print_function, absolute_import
import numpy as np
import selectivesearch
import cv2
import os
import random
import  config
import tools
import pickle
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt


def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=cv2.INTER_CUBIC):
    img = cv2.resize(in_image, (new_width, new_height), resize_mode)
    if out_image:
        cv2.imwrite(out_image, img)
    return img

def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return if_intersect
    if if_intersect:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter

def IOU(ver1, vertice2):
    vertice1 = [ver1[0], ver1[1], ver1[0]+ver1[2], ver1[1]+ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3], vertice2[0], vertice2[2], vertice2[1], vertice2[3])
    if area_inter:
        area_1 = ver1[2] * ver1[3]
        area_2 = vertice2[4] * vertice2[5]
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    return False

def clip_pic(img, rect):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    x1 = x + w
    y1 = y + h
    return img[y:y1, x:x1, :], [x, y, x1, y1, w, h]

def load_train_proposals(datafile, num_class, save_path, threshold=0.5, is_svm=False, save=False):
    f = open(datafile, 'r')
    train_list = f.readlines()
    for num, line in enumerate(train_list):
        labels = []
        images = []
        tmp = line.strip().split(' ')
        img = io.imread(tmp[0])

        img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9,
                                                            min_size=10)
        candidates = set()
        for r in regions:
            if r['rect'] in candidates:
                continue
            if r['size'] < 220:
                continue
            if (r['rect'][2] * r['rect'][3]) < 800:
                continue
            proposal_img, proposal_vertice = clip_pic(img, r['rect'])
            if len(proposal_img) == 0:
                continue
            x, y, w, h = r['rect']
            if w == 0 | h == 0 | x == 0 | y == 0:
                continue
            [a, b, c] = np.shape(proposal_img)
            if a == 0 or b == 0 or c == 0:
                continue
            resized_proposal_img = resize_image(proposal_img, config.IMAGE_SIZE, config.IMAGE_SIZE)
            candidates.add(r['rect'])

            # plt.imshow(img_float)
            # plt.figure()
            # plt.imshow(resized_proposal_img)
            ref_rect = tmp[2].split(',')
            ref_rect_int = [int(i) for i in ref_rect]
            iou_val = IOU(ref_rect_int, proposal_vertice)
            if iou_val == False:
                continue
            # img_float = np.asarray(resized_proposal_img / 255., dtype='float32')
            images.append(resized_proposal_img)
            index = int(tmp[1])
            if is_svm:
                if iou_val < threshold:
                    labels.append(0)
                else:
                    labels.append(index)
            else:
                label = np.zeros(num_class + 1)
                # print(iou_val)
                if iou_val < threshold:
                    label[0] = 1
                else:
                    label[index] = 1
                labels.append(label)
            # print('iou val = {}, threshold = {}'.format(iou_val, threshold))
            # print(labels)
            # cv2.imshow('im', img_float)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        # neg_count = 0
        # label_1_count = 0
        # label_2_count = 0
        # #expand image dataset
        # for i in range(len(labels)):
        #     if labels[i][0] == 1.:
        #         neg_count += 1
        #     if labels[i][1] == 1.:
        #         label_1_count += 1
        #     if labels[i][2] == 1.:
        #         label_2_count += 1
        # expand_label_1_count = neg_count - label_1_count
        # expand_label_2_count = neg_count - label_2_count
        # for i in range(len(labels)):
        #     label = labels[i]
        #     if label[1] == 1.:
        #         image = images[i]
        #         im = Image.fromarray(image, mode='RGB')
        #         out1 = im.transpose(Image.FLIP_LEFT_RIGHT)
        #         out2 = im.transpose(Image.FLIP_TOP_BOTTOM)
        #         out3 = im.rotate(45)
        #         out4 = im.rotate(60)
        #         arr1 = np.array(out1)
        #         arr2 = np.array(out2)
        #         arr3 = np.array(out3)
        #         arr4 = np.array(out4)
        #         images.append(arr1)
        #         labels.append([0, 1, 0])
        #         images.append(arr2)
        #         labels.append([0, 1, 0])
        #         images.append(arr3)
        #         labels.append([0, 1, 0])
        #         images.append(arr4)
        #         labels.append([0, 1, 0])
        #     if label[2] == 1.:
        #         image = images[i]
        #         im = Image.fromarray(image, mode='RGB')
        #         out1 = im.transpose(Image.FLIP_LEFT_RIGHT)
        #         out2 = im.transpose(Image.FLIP_TOP_BOTTOM)
        #         out3 = im.rotate(45)
        #         out4 = im.rotate(60)
        #         arr1 = np.array(out1)
        #         arr2 = np.array(out2)
        #         arr3 = np.array(out3)
        #         arr4 = np.array(out4)
        #         images.append(arr1)
        #         labels.append([0, 0, 1])
        #         images.append(arr2)
        #         labels.append([0, 0, 1])
        #         images.append(arr3)
        #         labels.append([0, 0, 1])
        #         images.append(arr4)
        #         labels.append([0, 0, 1])
        tools.view_bar("processing image of %s" % datafile.split('\\')[-1].strip(), num + 1, len(train_list))
        if save:
            np.save((os.path.join(save_path, tmp[0].split('/')[-1].split('.')[0].strip()) + '_data.npy'), [images, labels])
    print(' ')
    f.close()

def load_from_npy(data_set):
    images, labels = [], []
    data_list = os.listdir(data_set)
    for ind, d in enumerate(data_list):
        i, l = np.load(os.path.join(data_set, d), allow_pickle=True)
        # for im in i:
        #     plt.imshow(im)
        #     plt.show()
        images.extend(i)
        labels.extend(l)
        # cv2.imshow('im', i)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print('label {}'.format(l))
        tools.view_bar("load data of %s" % d, ind + 1, len(data_list))
    print(' ')
    return images, labels

if __name__ == '__main__':
    datafile = 'fine_tune_list.txt'
    load_train_proposals(datafile, num_class=17, save_path='save', is_svm=True)
