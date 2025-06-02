import tensorflow as tf
from PIL import Image
import sys
import numpy as np
import cv2
import copy
from argparse import ArgumentParser
import playsound
import subprocess
import time

ap = ArgumentParser(description='python main_cap.py')
ap.add_argument('--model', '-m', nargs='?', default='0', help='Specify loading file path of learned Model')
ap.add_argument('--gpu', '-g', type=int, default=-1, help='Specify GPU ID (negative value indicates CPU)')
ap.add_argument('--frame', '-f', type=int, default=3, help='Specify Grouping frame rate (default = 3)')
ap.add_argument('--method', '-d', type=int, default=1, help='Specify Method Flag (1 : Haarcascades Frontalface Default, 2 : Haarcascades Frontalface Alt1, 3 : Haarcascades Frontalface Alt2, Without : Haarcascades Frontalface Alt Tree)')

args = ap.parse_args()

if __name__ == '__main__':

    group = ['Gryffindor', 'Ravenclaw', 'Hufflpuff', 'Slytherin']

    if not args.model == '0':
        try:
            model = tf.keras.models.load_model('sorting_hat_classifier1.h5')
            print('Loading Model : ' + args.model)
        except:
            print('ERROR!!')
            print('Usage : Input File Path of Model (ex ./hoge.model)')
            sys.exit()

    if args.method == 1:
        face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
    elif args.method == 2:
        face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
    elif args.method == 3:
        face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt2.xml')
    else:
        face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt_tree.xml')
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    cnt = 0
    sorting = 0
    introp = 0
    while(True):

        if not sorting and not introp:
            subprocess.Popen(['mplayer',"sounds/sorting_hat_intro.mp3"])
            introp = 1
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            t1 = time.time()
            sorting = 1 #0:not sorting 1:sorting 
            soundp = 0 #0:not playing 1:playing
            timef = 0 #0:not done playing audio 1:done playing audio

        if key == ord('r'):
            sorting = 0

        ret, frame = cap.read()
        if ret == False:
            break
        else:
            if cnt % args.frame == 0:
                gImg = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gImg, 1.3, 5)
                if len(faces) > 0 and sorting and soundp==0:
                    for num in range(len(faces)):
                        cropImg = copy.deepcopy(frame[faces[num][1]:faces[num][1]+faces[num][3], faces[num][0]:faces[num][0]+faces[num][2]])
                        resizeImg = cv2.resize(cropImg, (100, 100))
                        resizeImg = np.array(resizeImg)
                        resizeImg = resizeImg[np.newaxis,:,:,:]
                        tfdata = tf.data.Dataset.from_tensor_slices(resizeImg)
                        image_resized = tfdata.map(lambda x: tf.image.resize(x, size=(64,64))/255.0)
                        pre = next(iter(image_resized.batch(1)))
                        preds = model(pre)

                        text = group[tf.argmax(preds[0])]
                        x, y, w, h = faces[num]
                        if tf.argmax(preds[0]) == 0:  # Gr
                            color = (0, 0, 255)
                            if soundp == 0:
                                subprocess.Popen(['mplayer',"sounds/Gryffindor.mp3"]) 
                            soundp = 1  
                        elif tf.argmax(preds[0]) == 1:  # Ra
                            color = (255, 0, 0)
                            if soundp == 0:
                                subprocess.Popen(['mplayer',"sounds/Ravenclaw.mp3"])   
                            soundp = 1  
                        elif tf.argmax(preds[0]) == 2:  # Hu
                            color = (0, 255, 255)
                            if soundp == 0:
                                subprocess.Popen(['mplayer',"sounds/Hufflepuff.mp3"])   
                            soundp = 1  
                        elif tf.argmax(preds[0]) == 3:  # Sl
                            color = (0, 255, 0) 
                            if soundp == 0:
                                subprocess.Popen(['mplayer',"sounds/Slytherin.mp3"])   
                            soundp = 1                                
                        color0 = (0, 0, 0)
                        text0 = ("Sorting...")
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color0, 2)
                        cv2.putText(frame, text0, (x+int(w/5), y+h+30), cv2.FONT_HERSHEY_COMPLEX, 1.0, color0, thickness=3)
                elif len(faces) > 0 and sorting and soundp:
                    x, y, w, h = faces[num]
                    t2 = time.time()
                    if t2-t1 > 8:
                        timef = 1
                    if timef == 0:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color0, 2)
                        cv2.putText(frame, text0, (x+int(w/5), y+h+30), cv2.FONT_HERSHEY_COMPLEX, 1.0, color0, thickness=3)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, text, (x+int(w/5), y+h+30), cv2.FONT_HERSHEY_COMPLEX, 1.0, color, thickness=3)

                cv2.imshow('Sorting Hat', frame)
            cnt += 1

        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()