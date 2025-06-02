import tensorflow as tf
import os, glob
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import time
import random
import copy
import math
import six
import os
import os.path as pt
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.io as skimage
from argparse import ArgumentParser
import playsound
random.seed(5)

classes = ["Gryffindor","Ravenclaw","Hufflepuff","Slytherin"]
num_classes = len(classes)

ap = ArgumentParser(description='python main.py')
ap.add_argument('--indir', '-i', nargs='?', default='.jpg', help='Specify input files directory training data')
ap.add_argument('--outdir', '-o', nargs='?', default='Result', help='Specify output files directory for create result and save model file')
ap.add_argument('--model', '-m', nargs='?', default='0', help='Specify loading file path of learned Model')
ap.add_argument('--imsize', '-s', type=int, default=100, help='Specify size of image')
ap.add_argument('--method', '-d', type=int, default=1, help='Specify Method Flag (1 : Haarcascades Frontalface Default, 2 : Haarcascades Frontalface Alt1, 3 : Haarcascades Frontalface Alt2, Without : Haarcascades Frontalface Alt Tree)')

args = ap.parse_args()
opbase = args.outdir
argvs = sys.argv

# Path Separator
psep = '/'
if (args.outdir[len(opbase) - 1] == psep):
    opbase = opbase[:len(opbase) - 1]
if not (args.outdir[0] == psep):
    if (args.outdir.find('./') == -1):
        opbase = './result/' + opbase
# Create Opbase
t = time.ctime().split(' ')
if t.count('') == 1:
    t.pop(t.index(''))
opbase = opbase + '_' + t[1] + t[2] + t[0] + '_' + t[4] + '_' + t[3].split(':')[0] + t[3].split(':')[1] + t[3].split(':')[2]
if not (pt.exists(opbase)):
    os.mkdir(opbase)
    print('Output Directory not exist! Create...')
print('Output Directory:', opbase)

if __name__ == '__main__':

    group = ['Gryffindor', 'Ravenclaw', 'Hufflpuff', 'Slytherin']

    img = skimage.imread(args.indir)
    if not args.model == '0':
        try:
            model = tf.keras.models.load_model('sorting_hat_classifier1.h5')
            print('Loading Model : ' + args.model)
            filename = opbase + psep + 'result.txt'
            f = open(filename, 'w')
            f.write('Loading Model : {}\n'.format(args.model))
            f.close()
        except:
            print('ERROR!!')
            print('Usage : Input File Path of Model (ex ./hoge.model)')
            sys.exit()

    preImg, text = [], []
    gImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if args.method == 1:
        face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
    elif args.method == 2:
        face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
    elif args.method == 3:
        face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt2.xml')
    else:
        face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt_tree.xml')
    faces = face_cascade.detectMultiScale(gImg, 1.3, 5)
    for num in range(len(faces)):
        cropImg = copy.deepcopy(img[faces[num][1]:faces[num][1]+faces[num][3], faces[num][0]:faces[num][0]+faces[num][2]])
        resizeImg = cv2.resize(cropImg, (args.imsize, args.imsize))
        resizeImg = np.array(resizeImg)
        resizeImg = resizeImg[np.newaxis,:,:,:]
        tfdata = tf.data.Dataset.from_tensor_slices(resizeImg)
        image_resized = tfdata.map(lambda x: tf.image.resize(x, size=(64,64))/255.0)
        pre = next(iter(image_resized.batch(1)))
        preds = model(pre)

        print(group[tf.argmax(preds[0])])
        text = group[tf.argmax(preds[0])]
        x, y, w, h = faces[num]
        if tf.argmax(preds[0]) == 0:
            color = (255, 0, 0)
        elif tf.argmax(preds[0]) == 1:
            color = (0, 0, 255)
        elif tf.argmax(preds[0]) == 2:
            color = (255, 255, 0)
        elif tf.argmax(preds[0]) == 3:
            color = (0, 255, 0)
        else:
            color = (0, 0, 0)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (int(x+w/5), y+h+30), cv2.FONT_HERSHEY_COMPLEX, 1.0, color, thickness=3)

    filename = opbase + psep + 'result.jpg'
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))