from icrawler.builtin import BingImageCrawler
import os
import cv2
import copy

# gryffindor = ["harry james potter","ron weasley","hermione granger","albus dumbledore","minerva mcgonagall","james potter",
#               "lily potter","remus lupin","sirius black","neville longbottom","oliver wood"]
# ravenclaw = ["helena ravenclaw","quirell","ollivander","sybill trelawney","filius flitwick","gideroy lockhart",
#              "cho chang","luna lovegood","xenophilius lovegood","myrtle warren"]
# hufflepuff = ["eldritch diggory","newton scamander","nymphadora tonks","cedric diggory","susan bones","theseus scamander",
#               "justin finch-fletchley","ernie mcmillan","pomona sprout","zacharias smith"]
# slytherin = ["draco malfoy","severus snape","tom riddle","bellatrix lestrange","narcissa malfoy","lucius malfoy",
#              "dolores umbridge","horace slughorn","leta restrange","vincent crabbe","gregory goyle"]

gryffindor = ["gryffindor","harry james potter","ron weasley","hermione granger","albus dumbledore","minerva mcgonagall","james potter",
              "lily potter","remus lupin","sirius black","neville longbottom","oliver wood"]
ravenclaw = ["ravenclaw","helena ravenclaw","quirell","ollivander","sybill trelawney","filius flitwick","gideroy lockhart",
             "cho chang","luna lovegood","xenophilius lovegood","myrtle warren"]
hufflepuff = ["hufflepuff","eldritch diggory","newton scamander","nymphadora tonks","cedric diggory","susan bones","theseus scamander",
              "justin finch-fletchley","ernie mcmillan","pomona sprout","zacharias smith"]
slytherin = ["slytherin","draco malfoy","severus snape","tom riddle","bellatrix lestrange","narcissa malfoy","lucius malfoy",
             "dolores umbridge","horace slughorn","leta restrange","vincent crabbe","gregory goyle"]

houses = [gryffindor,ravenclaw,hufflepuff,slytherin]

n=0
for house in houses:
    for keyword in house[1:]:
        n+=1
        #crawler = BingImageCrawler(storage = {'root_dir' : './image/original/' + house[0]})
        #crawler.crawl(keyword = keyword, max_num = 20, file_idx_offset=(n-1)*20)

def cropFace(opbase, path, imsize, method,house):
    dir = opbase + '/crop/' + house
    if not (os.path.exists(dir)):
        os.mkdir(dir)
    for p in path:
        img = cv2.imread(opbase + '/original/' + house + '/' + p)
        gImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if method == 1:
            face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
        elif method == 2:
            face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
        elif method == 3:
            face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt2.xml')
        else:
            face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt_tree.xml')
        faces = face_cascade.detectMultiScale(gImg, 1.3, 5)
        for num in range(len(faces)):
            cropImg = copy.deepcopy(img[faces[num][1]:faces[num][1]+faces[num][3], faces[num][0]:faces[num][0]+faces[num][2]])
            resizeImg = cv2.resize(cropImg, (imsize, imsize))
            filename = dir + '/' + p[:-4] + '_' + str(num + 1) + '.png'
            cv2.imwrite(filename, resizeImg)

"""cropFace("./image",os.listdir('./image/original/gryffindor'),100,1,"gryffindor")
cropFace("./image",os.listdir('./image/original/hufflepuff'),100,1,"hufflepuff")
cropFace("./image",os.listdir('./image/original/ravenclaw'),100,1,"ravenclaw")
cropFace("./image",os.listdir('./image/original/slytherin'),100,1,"slytherin")"""