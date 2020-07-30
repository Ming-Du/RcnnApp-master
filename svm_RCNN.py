from __future__ import division, print_function, absolute_import
import numpy as np
import selectivesearch
import os
from sklearn import svm
from sklearn.externals import joblib
from preprocessing_RCNN import *
import cv2
import tools
import config
from alexnet import *
import torch
from torchvision import transforms
from PIL import Image
from skimage import io

normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def image_proposal(img_path):
    img = io.imread(img_path)
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lbl,regions = selectivesearch.selective_search(img,
                                                       scale=500,
                                                       sigma=0.9,
                                                       min_size=10)
    candidates = set()
    images = []
    vertices = []
    for r in regions:
        if r['rect'] in candidates:
            continue
        if r['size'] < 220:
            continue
        if (r['rect'][2] * r['rect'][3]) < 500:
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
        images.append(resized_proposal_img)
        vertices.append(r['rect'])
    return images, vertices

def generate_single_svm_train(train_file):
    save_path = train_file.rsplit('.', 1)[0].strip()
    if len(os.listdir(save_path)) == 0:
        print('Reading %s\'s svm dataset' %train_file.split('\\')[-1])
        load_train_proposals(train_file, 2, save_path, threshold=0.3, is_svm=True, save=True)
    print('rescoring svm dataset')
    images, labels = load_from_npy(save_path)
    return images, labels

def train_svm(train_filefolder, model):
    files = os.listdir(train_filefolder)
    svms = []
    for train_file in files:
        if train_file.endswith('.txt'):
            X, Y = generate_single_svm_train(os.path.join(train_filefolder, train_file))
            train_features = []
            for index, i in enumerate(X):
                with torch.no_grad():
                    data = Image.fromarray(i, mode='RGB')
                    data = transform(data)
                    # print(data.shape)
                    data = data.unsqueeze(0)

                    feats = model(data)
                    # print(model)
                    train_features.append(feats[0].numpy())
                    tools.view_bar("extract features of %s" % train_file, index + 1, len(X))
            print(' ')
            print('feature dimension')
            # print(np.shape(train_features))
            clf = svm.LinearSVC()
            print('fit svm')
            clf.fit(train_features, Y)
            svms.append(clf)
            joblib.dump(clf, os.path.join(train_filefolder, str(train_file.split('.')[0]) + '_svm.pkl'))
    return svms

def start():
    train_filefolder = config.TRAIN_SVM
    img_path = 'image_1306.jpg'
    imgs, verts = image_proposal(img_path)
    # tools.show_rect(img_path, verts)
    model = Alexnet(num_classes=3)
    model.load_state_dict(torch.load('./pre_train_model/find_tune_flower.pth'))
    model.classifier[6] = nn.Linear(4096, 4096)
    svms = []
    model.eval()
    for file in os.listdir(train_filefolder):
        if file.split('_')[-1] == 'svm.pkl':
            svms.append(joblib.load(os.path.join(train_filefolder, file)))
    if len(svms) == 0:
        svms = train_svm(train_filefolder, model)
    print('Done fitting svms')
    features = []
    with torch.no_grad():
        for img in imgs:
            data = Image.fromarray(img, mode='RGB')
            data = transform(data)
            data = data.unsqueeze(0)
            feat = model(data)
            features.append(feat[0].numpy())
    results = []
    results_label = []
    count = 0
    for f in features:
        for svm in svms:
            pred = svm.predict([f.tolist()])
            if pred[0] != 0:
                results.append(verts[count])
                results_label.append(pred[0])
        count = count + 1
    print('result')
    print(results)
    print('result label')
    print(results_label)
    tools.show_rect(img_path, results)

if __name__ == '__main__':
    start()
plt.show()