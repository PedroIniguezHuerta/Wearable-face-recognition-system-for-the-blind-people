############################################################
# if camera is not working run
# sudo modprobe bcm2835-v4l2
############################################################
#!/usr/bin/env python

import numpy as np
import time
import cv2
import os
import sqlite3
import datetime
from PIL import Image
import RPi.GPIO as GPIO

def singleton(cls):
    instance=cls()
    cls.__new__ = cls.__call__= lambda cls: instance
    cls.__init__ = lambda self: None
    return instance
 
@singleton
class Keypad():
    MATRIX = [ [1,   2,  3,   "A"],
               [4,   5,  6,   "B"],
               [7,   8,  9,   "C"],
               ["*", 0,  "#", "D"]]
    ROW    = [11, 12, 13, 15]
    COLUMN = [16, 18, 22, 7 ]
     
    def __init__(self):
        GPIO.setmode(GPIO.BOARD)
        self.press_time = 0
        self.read_again = True

    def prepareKeypadForRowReading(self):
        for j in range(len(self.COLUMN)):
            GPIO.setup(self.COLUMN[j], GPIO.OUT)
            GPIO.output(self.COLUMN[j], GPIO.LOW)
         
        for i in range(len(self.ROW)):
            GPIO.setup(self.ROW[i], GPIO.IN, pull_up_down=GPIO.PUD_UP)
     
    def prepareKeypadForColumnReading(self, row):
        for j in range(len(self.COLUMN)):
            GPIO.setup(self.COLUMN[j], GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
         
        GPIO.setup(self.ROW[row], GPIO.OUT)
        GPIO.output(self.ROW[row], GPIO.HIGH)

    def readKey(self):
        self.prepareKeypadForRowReading()
 
        for i in range(len(self.ROW)):
            row = GPIO.input(self.ROW[i])
            if row == 0 and self.read_again == True:
                self.press_time = time.time()
                self.read_again = False
                self.prepareKeypadForColumnReading(row)

                for j in range(len(self.COLUMN)):
                    if GPIO.input(self.COLUMN[j]) == 1:
                        self.resetPins()
                        return self.MATRIX[i][j]

        self.resetPins()
        if time.time() - self.press_time > 1:
            self.read_again = True

        return -1
         
    def resetPins(self):
        for i in range(len(self.ROW)):
            GPIO.setup(self.ROW[i], GPIO.IN, pull_up_down=GPIO.PUD_UP) 
        for j in range(len(self.COLUMN)):
            GPIO.setup(self.COLUMN[j], GPIO.IN, pull_up_down=GPIO.PUD_UP)

@singleton
class GCo:
    DNN                  = 0
    HAAR_CASCADE         = 1
    LBP                  = 2

    LBP_RECOGNITION      = 0
    EIGEN_RECOGNITION    = 1
    FISHER_RECOGNITION   = 2

    MAX_REG_PICTURES     = 5
    GRIDX_SIZE           = 300
    GRIDY_SIZE           = 300
    DISPLAYX_SIZE        = 640
    DISPLAYY_SIZE        = 480
    SKIP_FRAMES          = 1500  # 150 (3 sec camera, 1 sec face detection)
                          #  50 (1 sec camera, 1 sec face detection)
    CONFIDENCE_GRADE     = 0.3

    WORKING_DIRECTORY    = "./"
    DATASET_DIR          = "dataset/"
    TEMP_DIR             = "dataset_temp_dir/"
    FACES_DIR            = "facesDetected/"
    IMAGES_BEF_PROCESS   = "imagesBeforeProcessed/"

    DATABASE_FILE        = "database.db"

    LBP_ENABLED          = True
    EIGEN_ENABLED        = False
    FISHER_ENABLED       = False

    LBP_TRAINING_FILE    = "recognizer/LBPData.yml"
    EIGEN_TRAINING_FILE  = "recognizer/EigenData.yml"
    FISHER_TRAINING_FILE = "recognizer/FisherData.yml"
    PROTOTXT_FILE        = "deploy.prototxt"

    CLASSIFIER_FILE      = "haarcascade_frontalface_default.xml"
    LBP_CLASSIFIER_FILE  = "lbpcascade_frontalface.xml"
    CAFFEMODEL_FILE      = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    CAMERAFIX_FILE       = "/home/pi/camerafix.sh"

    AUDIONAMES           = "names/"
    AUDIOCOMMANDS        = "commands/"
    AUDIOPEOPLE          = "people/"
    AUDIODISTANCES       = "distances/"
    AUDIOPLAYER          = "aplay"
    AUDIORECORDER        = "python recorder.py "

    AU_DETECTION         = "detection"
    AU_RECOGNITION       = "recognition"
    AU_CAMERA            = "camera"
    AU_METHOD            = "method"
    AU_CAPTURENOW        = "capturenow"
    AU_REGISTRATION      = "registration"
    AU_CAMERAONOFF       = "CamaraONOFF"

    AU_LAST              = "last"
    AU_NEXT              = "next"
    AU_ENABLEDISABLE     = "enabledisable"
    AU_PLAYMUSIC3        = "playmusic3"
    AU_PLAYMUSIC2        = "playmusic2"
    AU_PLAYMUSIC         = "playmusic"
    AU_ENTER             = "enter"
    AU_WRITEIMAGE        = "writeimage"
    AU_LANGUAGE          = "language"
    AU_CAMARADISABLED    = "camaradisabled"
    AU_CAMARAENABLED     = "camaraenabled"
    AU_LBPMODE           = "lbpmode"
    AU_DNNMODE           = "dnnmode"
    AU_HAARCASCADEMODE   = "haarcascademode"
    AU_DETECTIONDISABLED = "detectiondisabled"
    AU_DETECTIONENABLED  = "detectionenabled"
    AU_RECOGNITIONDISABLED = "recognitiondisabled"
    AU_RECOGNITIONENABLED = "recognitionenabled"
    AU_UNKNOWNUSER       = "unknownuser"
    AU_ASKNAME           = "askName"
    AU_PIII              = "piii"
    AU_CONFIRMNAME       = "confirmName"
    AU_NAMEREGISTERED    = "nameRegistered"
    AU_REGISTRATIONSTARTS = "registratinstarts"
    AU_REGISTRATIONABORTED = "registrationAborted"
    AU_TAKEPHOTO         = "takephoto"
    AU_LOADING           = "loading"
    AU_READY             = "ready"
    AU_CAPTURESTART      = "captureStart"
    AU_CAPTUREDONE       = "captureDone"
    AU_IMAGEWRITTEN      = "imagewritten"
    AU_FACEDETECTEDLBP   = "facedetectedlbp"
    AU_FACEDETECTEDHC    = "facedetectedhc"
    AU_FACEDETECTEDDNN   = "facedetecteddnn"
    AU_NOFACES           = "nofaces"
    AU_QUIT              = "quit"
    AU_LEAVING           = "leaving"
    AU_DETECTIONDONE     = "detectiondone"
    AU_RECOGNITIONDONE   = "recognitiondone"
    AU_PEOPLERECOGNIZED  = "peoplerecognized"
    AU_PHOTONOW          = "photonow"
    AU_WELCOME           = "welcome"

    AU_0                 = "0"
    AU_1                 = "1"
    AU_2                 = "2"
    AU_3                 = "3"
    AU_4                 = "4"
    AU_5                 = "5"
    AU_6                 = "6"
    AU_7                 = "7"
    AU_8                 = "8"
    AU_9                 = "9"

    IMAGE_TITLE          = "frame"
    NM_OK                = 0
    NM_STOPMENU          = 1
    NM_TRAINNOW          = 2
    NM_TRIGGERNOW        = 3
    NM_WRITEIMAGE        = 4
    NM_QUIT              = 5


@singleton
class Camera:
    def __init__(self):
        self._camera = None
        self._number = 0
        self._enabled = False
        self._initializeHardware()

    def _initializeHardware(self):
        os.system(GCo.CAMERAFIX_FILE) 

    def _adjustSettings(self):
        # Configure exposure time
        #cap.set(cv2.CAP_PROP_EXPOSURE, 400)
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, GCo.DISPLAYX_SIZE)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, GCo.DISPLAYY_SIZE)
        # self._camera.set(cv2.cv.CV_CAP_PROP_EXPOSURE, -6.0)
        # self._camera.set(cv2.cv.CV_CAP_PROP_GAIN, 4.0)
        # self._camera.set(cv2.cv.CV_CAP_PROP_BRIGHTNESS, 144.0)
        # self._camera.set(cv2.cv.CV_CAP_PROP_CONTRAST, 27.0)
        # self._camera.set(cv2.cv.CV_CAP_PROP_HUE, 13.0) # 13.0
        # self._camera.set(cv2.cv.CV_CAP_PROP_SATURATION, 28.0)

    def _warmUp(self):
        # allow some time to warm up camera
        print("Warming up camera")
        time.sleep(2.0)
        print("Warming up done")
    
    def isEnabled(self):
        return self._enabled

    def setEnabled(self, enabled):
        self._enabled = enabled
        self.close()
        if enabled:
           self.start(self._number)
           

    def read(self):
        return self._camera.read()
    
    def start(self, number=0):
        self._number = number
        self._camera = cv2.VideoCapture(self._number)
        self._adjustSettings()
        self._warmUp()
        return self._camera

    def stop(self):
        cv2.VideoCapture(self._number).release()

    def close(self):
#       cv2.destroyAllWindows()
        self._camera.release()

@singleton
class AudioController:
    def __init__(self):
        self._path = GCo.WORKING_DIRECTORY
        self._languages = ["espanol", "english"]
        self._current_language = 0

    def getCurrentLanguage(self):
        return self._current_language

    def setCurrentLanguage(self, current_language):
        self._current_language = self.current_language

    def setNextLanguage(self):
        self._current_language += 1 
        if self._current_language >= len(self._languages):
            self._current_language = 0

    def setPreviousLanguage(self):
        self._current_language -= 1
        if self._current_language < 0:
            self._current_language = len(self._languages) - 1

    def cmdName(self, name, wait = False):
        print("name=",name)
        async = "&"
        if wait:
            async=""
        os.system(GCo.AUDIOPLAYER + " " + self._path + GCo.AUDIONAMES + name + ".wav" + async)

    def cmdSound(self, command, wait = False):
        async = "&"
        if wait:
            async=""
        os.system(GCo.AUDIOPLAYER + " " + self._path + GCo.AUDIOCOMMANDS + self._languages[self._current_language] + "/" + command + ".wav" + async)

    def cmdPeopleDetected(self, detected_people, wait=True):
        async = "&"
        if wait:
            async=""
        os.system(GCo.AUDIOPLAYER + " " + self._path + GCo.AUDIOPEOPLE + self._languages[self._current_language] + "/" + str(detected_people) + "people.wav" + async)

    def cmdDistance(self, name, distance, wait = True):
        async = "&"
        if wait:
            async=""

        # play name of the user
        self.cmdName(name,True)

        strdist=str(int(round(distance%100/10.0))*10)
        if distance >= 400:
            strdist = str(distance/100) + "m"
        elif distance < 95:
            strdist += "cm"
        else:
            strdist = str(distance/100) + "m" + strdist + "cm"

        print(strdist)
        #play distance of the user
        os.system(GCo.AUDIOPLAYER + " " + self._path + GCo.AUDIODISTANCES + self._languages[self._current_language] + "/" + strdist + ".wav" + async)


@singleton
class ImageManagement():
    def __init__(self):
        self._path = GCo.WORKING_DIRECTORY
        self._audioctl = AudioController()

    def setHeader(self, frame, text, location):
        cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255))

    def setStatus(self, frame, text, location):
        cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))

    def writeImage(self, frame, path, play_sound=True):
        xfile = 'image{:%Y-%m-%d %H:%M:%S}.jpg'.format(datetime.datetime.now())
        filename = path + xfile.replace(" ","").replace(":","").replace("-","")
        cv2.imwrite(filename, frame)
        if play_sound:
            self._audioctl.cmdSound(GCo.AU_IMAGEWRITTEN, wait=True)
        return filename

    def calcDistance(self, x, y, h):
        distance = (640/h)*20
    #    rx = x # reference on x
    #    ry = y # reference on y
    #    offset = arctan(abs(rx-x)/distance)
        return distance

    def markFace(self, frame, audio, name, id, det_conf, conf, x, y, h, color, position):
        distance = self.calcDistance(x,y,h)
        text = name + str(id) + ":" + "{:02d}%".format(int(conf)) +  ", " + str(distance) + "cm"
        line = 15
        if position == GCo.LBP_RECOGNITION:
            text = "lbp:" + text
        elif position == GCo.EIGEN_RECOGNITION:
            text = "Eig:" + text
            line = 30
        elif position == GCo.FISHER_RECOGNITION:
            text = "Fis:" + text
            line = 45

        cv2.putText(frame, text, (x+2,y+h+line), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color ,2)
        
        if audio:
           if conf < GCo.CONFIDENCE_GRADE:
               self._audioctl.cmdSound(GCo.AU_UNKNOWNUSER, wait=True)
           name = name.replace("hc","").replace("dnn","")
           self._audioctl.cmdDistance(name,distance,wait=True)

@singleton
class DataBase():
    def __init__(self):
        self._cursor = None
        self._dbname = GCo.DATABASE_FILE
        self._cursor = None
        self._path = GCo.WORKING_DIRECTORY
        self._conn = None

    def _openDB(self):
        self._conn = sqlite3.connect(self._path + self._dbname)
        self._cursor = self._conn.cursor()

    def _closeDB(self):
        self._conn.commit()
        self._conn.close()

    def getNamesFromIds(self, ids):
        self._openDB()
        print("ids:",str(ids))
        self._cursor.execute("select name from users where id = (?);", (ids,))
#       self._cursor.execute("select name from users where id in (" + str(ids) + ");")
        result = self._cursor.fetchall()
        self._closeDB()
        return result

    def recreateDB(self):
        print("Destruyendo y recreando base de datos")
        os.system("rm " + self._path + self._dbname)

        os.system("rm " + self._path + GCo.LBP_TRAINING_FILE)
        os.system("rm " + self._path + GCo.EIGEN_TRAINING_FILE)
        os.system("rm " + self._path + GCo.FISHER_TRAINING_FILE)

        os.system("touch " + self._path + GCo.LBP_TRAINING_FILE)
        os.system("touch " + self._path + GCo.EIGEN_TRAINING_FILE)
        os.system("touch " + self._path + GCo.FISHER_TRAINING_FILE)

        self._openDB()
        sql = """
        DROP TABLE IF EXISTS users;
        CREATE TABLE users (id integer unique primary key autoincrement, name text);
        """
        self._cursor.executescript(sql)
        self._closeDB()
        print("Base de datos fue reconstruida")

    def addUser(self, user):
        self._openDB()
        self._cursor.execute('INSERT INTO users (name) VALUES (?)', (user,))
        id = self._cursor.lastrowid
        self._closeDB()
        print("id=",id)
        return id

    def getNextId(self):
        self._openDB()
        id = self._cursor.lastrowid
        self._closeDB()
        return id


class NavigationMenu():
    def __init__(self):
        self._camera = Camera()
        self._detection_enabled = False
        self._recognition_enabled = False
        self._registration_mode = False
        self._force_display = False
        self._gHcName = None
        self._gDnnName = None
        self._trigger_flag = False
        self._detection_method = GCo.HAAR_CASCADE

        self._frame = None
        self._commands = ["detection","recognition","camara","method","trigger", "registration"]
        self._current_command = 0
        self._path = GCo.WORKING_DIRECTORY
        self._audioctl = AudioController()

    def _disableAll(self):
        self._detection_enabled = False
        self._recognition_enabled = False
        self._force_display = False

    def isDetectionEnabled(self):
        return self._detection_enabled
    
    def isRecognitionEnabled(self):
        return self._recognition_enabled
    
    def isRegistrationMode(self):
        return self._registration_mode
    
    def isForceDisplayEnabled(self):
        return self._force_display

    def isTriggerFlagEnabled(self):
        return self._trigger_flag

    def getDetectionMethod(self):
        return self._detection_method
    
    def setDetectionEnabled(self, enabled):
        self._detection_enabled = enabled
    
    def setRecognitionEnabled(self, enabled):
        self._recognition_enabled = enabled
    
    def setRegistrationMode(self, enabled):
        self._registration_mode = enabled
    
    def setForceDisplayEnabled(self, enabled):
        self._force_display = enabled

    def setTriggerFlagEnabled(self, enabled):
        self._trigger_flag = enabled

    def setDetectionMethod(self, method):
        self._detection_method = method

    def nextCommand(self):
        self._current_command += 1
        if self._current_command >= len(self._commands):
            self._current_command = 0
        print("Current command=", self._commands[self._current_command])
        if self._commands[self._current_command] == "detection":
            self._audioctl.cmdSound(GCo.AU_DETECTION, wait=True)
        elif self._commands[self._current_command] == "recognition":
            self._audioctl.cmdSound(GCo.AU_RECOGNITION, wait=True)
        elif self._commands[self._current_command] == "camara":
            self._audioctl.cmdSound(GCo.AU_CAMERA, wait=True)
        elif self._commands[self._current_command] == "method":
            self._audioctl.cmdSound(GCo.AU_METHOD, wait=True)
        elif self._commands[self._current_command] == "trigger":
            self._audioctl.cmdSound(GCo.AU_CAPTURENOW, wait=True)
        elif self._commands[self._current_command] == "registration":
            self._audioctl.cmdSound(GCo.AU_REGISTRATION, wait=True)
        elif self._commands[self._current_command] == "camara":
            self._audioctl.cmdSound(GCo.AU_CAMARAONOFF, wait=True)

    def previousCommand(self):
        self._current_command -= 1
        if self._current_command < 0:
            self._current_command = len(self._commands) - 1
        print("Current command=", self._commands[self._current_command])
        self._audioctl.cmdSound(GCo.AU_LAST, wait=True)

    def processCommand(self):
        print("Processing command=", self._commands[self._current_command])
        self._audioctl.cmdSound(GCo.AU_ENABLEDISABLE, wait=True)

    def playMusic(self):
        self._audioctl.cmdSound(GCo.AU_PLAYMUSIC3)

    def playIntro(self, frame):
        self._audioctl.cmdSound(GCo.AU_0, wait=True)
        if self.processKey(frame, 1, 1):
            return
        self._audioctl.cmdSound(GCo.AU_1, wait=True)
        if self.processKey(frame, 1, 1):
            return
        self._audioctl.cmdSound(GCo.AU_2, wait=True)
        if self.processKey(frame, 1, 1):
            return
        self._audioctl.cmdSound(GCo.AU_3, wait=True)
        if self.processKey(frame, 1, 1):
            return
        self._audioctl.cmdSound(GCo.AU_4, wait=True)
        if self.processKey(frame, 1, 1):
            return
        self._audioctl.cmdSound(GCo.AU_5, wait=True)
        if self.processKey(frame, 1, 1):
            return
        self._audioctl.cmdSound(GCo.AU_6, wait=True)
        if self.processKey(frame, 1, 1):
            return
        self._audioctl.cmdSound(GCo.AU_7, wait=True)
        if self.processKey(frame, 1, 1):
            return
        self._audioctl.cmdSound(GCo.AU_8, wait=True)
        if self.processKey(frame, 1, 1):
            return
        self._audioctl.cmdSound(GCo.AU_9, wait=True)
        if self.processKey(frame, 1, 1):
            return
        self._audioctl.cmdSound(GCo.AU_WRITEIMAGE, wait=True)
        if self.processKey(frame, 1, 1):
            return
        self._audioctl.cmdSound(GCo.AU_NEXT, wait=True)
        if self.processKey(frame, 1, 1):
            return
        self._audioctl.cmdSound(GCo.AU_LAST, wait=True)
        if self.processKey(frame, 1, 1):
            return
        self._audioctl.cmdSound(GCo.AU_ENTER, wait=True)

    def changeLanguage(self):
        self._audioctl.setNextLanguage()
        self._audioctl.cmdSound(GCo.AU_LANGUAGE)

    def cameraONOFF(self):
        self._disableAll()
        if self._camera.isEnabled():
            print("Camera and detection Disabled")
            self._force_display = True
            self._camera.setEnabled(False)
            self._audioctl.cmdSound(GCo.AU_CAMARADISABLED, wait=True)
        else:
            print("Camera Enabled")
            self._camera.setEnabled(True)
            self._audioctl.cmdSound(GCo.AU_CAMARAENABLED, wait=True)

    def switchMethod(self):
        if self._detection_method == GCo.HAAR_CASCADE:
            self._detection_method = GCo.DNN
            self._audioctl.cmdSound(GCo.AU_DNNMODE, wait=True)
        elif self._detection_method == GCo.DNN:
            self._detection_method = GCo.LBP
            self._audioctl.cmdSound(GCo.AU_LBPMODE, wait=True)
        else: # LBP Mode
            self._detection_method = GCo.HAAR_CASCADE
            self._audioctl.cmdSound(GCo.AU_HAARCASCADEMODE, wait=True)

    def triggerNow(self):
        self._trigger_flag = True
        self._camera.setEnabled(True)
        if self._recognition_enabled:
            if self._recognition_enabled == False:
                self.recognitionONOFF()
        else:
            if self._detection_enabled == False:
                self.detectionONOFF()

    def detectionONOFF(self):
        self._HcName = None
        self._DnnName = None

        if self._detection_enabled and self._recognition_enabled:
            print("Face Detection only")
            self._recognition_enabled = False
            self._camera.setEnabled(False)
        elif self._detection_enabled:
            print("Face Detection disabled")
            self._camera.setEnabled(False)
            self._detection_enabled = False
            self._grecognition_enabled = False
            self._audioctl.cmdSound(GCo.AU_DETECTIONDISABLED, wait=True)
        else:
            print("Face Detection and camera enabled")
            self._camera.setEnabled(True)
            self._recognition_enabled = False
            self._detection_enabled = True
            self._audioctl.cmdSound(GCo.AU_DETECTIONENABLED, wait=True)

    def recognitionONOFF(self):
        if self._recognition_enabled:
            print("Face Detection and Recognition = OFF")
            self._recognition_enabled = False
            self._detection_enabled = False 
            self._camera.setEnabled(False)
            self._audioctl.cmdSound(GCo.AU_RECOGNITIONDISABLED, wait=True)
        else:
            print("Face Recognition = ON")
            self._camera.setEnabled(True)
            self._detection_enabled = True 
            self._recognition_enabled = True
            self._audioctl.cmdSound(GCo.AU_RECOGNITIONENABLED, wait=True)

    def processKey(self, frame, waittime, level):
        key = cv2.waitKey(waittime)
        kpkey = Keypad().readKey()
        if key == -1 and kpkey == -1:
            return GCo.NM_OK

        print("key=",key)
        if key == ord("j") or key == ord("0") or key == 176 or kpkey == 0: # braile j=0
            return GCo.NM_QUIT
        elif key == ord("+") or key == 171 or kpkey == "A":
            self._nextCommand()
        elif key == ord("-") or key == 173 or kpkey == "B":
            self.previousCommand()
        elif key == 13 or key == 141 or kpkey == "D":
            self.processCommand()
        elif key == ord("a") or key == ord("1") or key == 177 or kpkey == 1: # braile a=1
            return GCo.NM_TRAINNOW
        elif key == ord("b") or key == ord("2") or key == 178 or kpkey == 2: # braile b=2
            self.playMusic()
        elif key == ord("c") or key == ord("3") or key == 179 or kpkey == 3: # braile c=3
            self.cameraONOFF()
        elif key == ord("d") or key == ord("4") or key == 180 or kpkey == 4: # braile d=4
            self.switchMethod()
        elif key == ord("e") or key == ord("5") or key == 181 or kpkey == 5: # braile e=5
            self.triggerNow()
            return GCo.NM_TRIGGERNOW
        elif key == ord("f") or key == ord("6") or key == 182 or kpkey == 6: # braile f=6
            if level == 0:
                self.playIntro(frame)
            else:
                return GCo.NM_STOPMENU
        elif key == ord("g") or key == ord("7") or key == 183 or kpkey == 7: # braile g=7
            self.detectionONOFF()
        elif key == ord("h") or key == ord("8") or key == 184 or kpkey == 8: # braile h=8
            self.changeLanguage()
        elif key == ord("i") or key == ord("9") or key == 185 or kpkey == 9: # braile i=9
            self.recognitionONOFF()
        elif key == ord("w") or key == ord("*") or key == 170 or kpkey == "*":
            print("just to write image")
            return GCo.NM_WRITEIMAGE

        return GCo.NM_OK

class Registration():
    def __init__(self):
        self._path = GCo.WORKING_DIRECTORY

        self._audioctl = AudioController()
        self._id = 0
        self._counter = 0
        self._user = "U"
        self._username = "U"

    def _captureName(self, frame):
        self._username = self._getNextUserName()

        while True:
            self._audioctl.cmdSound(GCo.AU_ASKNAME, wait=True)
            self._audioctl.cmdSound(GCo.AU_PIII, wait=True)
            name = GCo.AUDIONAMES + self._username
            os.system(GCo.AUDIORECORDER + name + " 2")
            self._audioctl.cmdSound(GCo.AU_CONFIRMNAME, wait=True)
            
            while True:
                cv2.imshow(GCo.IMAGE_TITLE, frame)
                key = cv2.waitKey(50)
                kpkey = Keypad().readKey()
                if key != -1 or kpkey != -1:
                    break;

            if key == ord("a") or key == ord("1") or key == 177 or kpkey == 1:
                break;
            elif key == ord("j") or key == ord("0") or key == 176 or kpkey == 0:
                return ""

        self._audioctl.cmdSound(GCo.AU_NAMEREGISTERED, wait=True)
        return self._username

    def getUserName(self):
        return self._username

    def _getNextUserName(self):
        username = self._user
        while True:
            username = self._user + str(self.getNextId())
            if os.path.isfile(GCo.AUDIONAMES + username + ".wav") == False:
                break;
        return username

    def getNextId(self):
        self._counter = 0
        self._id += 1
        return self._id

    def getCurrentId(self):
        return self._id

    def getNextCounter(self):
        self._counter += 1
        return self._counter

    def startRegistration(self, frame):
        print("Starting registration")
        os.system("rm " + self._path + GCo.TEMP_DIR + "*.jpg")
        self._audioctl.cmdSound(GCo.AU_REGISTRATIONSTARTS, wait=True)
        username = self._captureName(frame)
        if username == "":
            print("registration aborted")
            self._audioctl.cmdSound(GCo.AU_REGISTRATIONABORTED, wait=True)
        else:
            self._audioctl.cmdSound(GCo.AU_TAKEPHOTO, wait=True)
            
            while True:
                cv2.imshow(GCo.IMAGE_TITLE, frame)
                key = cv2.waitKey(50)
                kpkey = Keypad().readKey()

                if key ==  ord('a') or key == ord("1") or key == 177 or kpkey == 1:
                    break;
                elif key == ord("j") or key == ord("0") or key == 176 or kpkey == 0:
                    self._audioctl.cmdSound(GCo.AU_REGISTRATIONABORTED, wait=True)
                    return ""

            self._audioctl.cmdSound(GCo.AU_CAPTURESTART, wait=True)

        return username

    def stopRegistration(self):
        print("Stopping registration")
        self._audioctl.cmdSound(GCo.AU_CAPTUREDONE, wait=True)


class Trainer():
    def __init__(self):
        self._db = DataBase()
        self._path = GCo.WORKING_DIRECTORY
        self._dataset = self._path + 'dataset'
        self._LBPRecognizer = cv2.face.LBPHFaceRecognizer_create()
        self._EigenRecognizer = cv2.face.EigenFaceRecognizer_create()
        self._FisherRecognizer = cv2.face.FisherFaceRecognizer_create()

    def _datasetRebuild(self):
        dataset_directory = [os.path.join(self._dataset,f) for f in os.listdir(self._dataset)]
        faces = []
        ids = []
        workdir = GCo.TEMP_DIR
        os.system("mkdir " + workdir)
        for image_file in dataset_directory:
            faceImg = Image.open(image_file).convert('L')
            faceNp = np.array(faceImg,'uint8')
            faceNp = cv2.resize(faceNp,(GCo.GRIDX_SIZE, GCo.GRIDY_SIZE))
            faces.append(faceNp)

            #cv2.imshow("FaceDemo", faceNp)
            #key = cv2.waitKey(200)

            #print(image_file)
            filename = os.path.split(image_file)[-1].split('.')

            id = self._db.addUser(filename[0])
            ids.append(id)
            new_name = workdir + filename[0] + "." + str(id) + "." + filename[2]
            #print("oldname: " + image_file + ". new name: " + new_name)
            os.system("mv " + image_file + " " + new_name)

        #cv2.imclose("FaceDemo")
        #cv2.destroyAllWindows()
            
        os.system("mv " + workdir + "* " + self._dataset)
#        os.system("rmdir " + workdir)

        return np.array(ids), faces

    def addUserToDB(self, user, uid):
        for lcounter in range (1, GCo.MAX_REG_PICTURES+1):
            os.system("mv " + GCo.TEMP_DIR + user + "dnn.%d.jpg dataset/%s.%d.jpg"%(lcounter,user+"dnn",uid))
            os.system("mv " + GCo.TEMP_DIR + user + "hc.%d.jpg dataset/%s.%d.jpg"%(lcounter,user+"hc",uid))
            os.system("mv " + GCo.TEMP_DIR + user + "lbp.%d.jpg dataset/%s.%d.jpg"%(lcounter,user+"lbp",uid))
        return 0

    def trainAll(self):
        self._db.recreateDB()
        print("reentrenando base de datos")
        ids, faces = self._datasetRebuild()
        print(len(ids), len(faces))

        if GCo.EIGEN_ENABLED:
            start = time.time()
            print("Eigen training starts")
            self._EigenRecognizer.train(faces, ids)
            self._EigenRecognizer.save(self._path + GCo.EIGEN_TRAINING_FILE)
            print("Eigen training done")
            end = time.time()
            print("Eigen Training time=", end-start)

        if GCo.FISHER_ENABLED:
            start = time.time()
            start = time.time()
            self._FisherRecognizer.train(faces, ids)
            self._FisherRecognizer.save(self._path + GCo.FISHER_TRAINING_FILE)
            print("Fisher training done")
            end = time.time()
            print("Fisher Training time=", end-start)

        if GCo.LBP_ENABLED:
            start = time.time()
            print("LBP training starts")
            self._LBPRecognizer.train(faces, ids)
            self._LBPRecognizer.save(self._path + GCo.LBP_TRAINING_FILE)
            print("LBP training done")
            end = time.time()
            print("LBP Training time=", end-start)

        print("base de datos reentrenada")

class BaseDetector():
    def __init__(self):
        self.path = GCo.WORKING_DIRECTORY
        self.im = ImageManagement()
        self.faces = None
        self.people_detected = 0
        self.color = (0, 255, 0)
        self.regFaces = []
        self.audio = False
        self._dataset_dir = self.path + GCo.DATASET_DIR
        self._temp_dir = self.path + GCo.TEMP_DIR

    def isDetectorValid(self):
        pass

    def initialize(self):
        pass

    def applyDetections(self, frame):
        pass

    def registerDetections(self, frame, gray, username, id):
        pass

    def saveFaceToFile(self, frame, x, y, w, h):
        face = cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
        filename = self.im.writeImage(face, self.path + GCo.FACES_DIR, play_sound=False)

    def clearRegistrations(self):
        self.regFaces = []

    def getRegistrationsNumber(self):
        return len(self.regFaces)

class HaarCascadeDetector(BaseDetector):
    def __init__(self):
        BaseDetector.__init__(self)
        self._classifier_file = self.path + GCo.CLASSIFIER_FILE
        self._face_cascade = None
        self.initialize()

    def isDetectorValid(self):
        return (self._face_cascade != None)

    def initialize(self):
        os.system("mkdir " + self._temp_dir)
        self.color = (0, 255, 0)
        if os.path.isfile(self._classifier_file):
            self._face_cascade = cv2.CascadeClassifier(self._classifier_file)

    def faceDetection(self, frame, gray):
        start = time.time()
        self.people_detected = 0
        self.audio = True
        self.faces = self._face_cascade.detectMultiScale(gray, 1.3, 5)
        end = time.time()
        print("HAAR CASCADE Face Detection time=", end-start)
        return self.faces

    def applyDetections(self, frame):
        self.people_detected = 0

        if self.faces == None:
            return 0

        for (x,y,w,h) in self.faces:
            confidence = 100
            self.people_detected += 1

            ###########################################
            # Save images to file
            ###########################################
            self.saveFaceToFile(frame, x, y, w, h)

            ###########################################
            # Draw the box and percentage around face
            ###########################################
            cv2.rectangle(frame, (x,y), (x+w,y+h), self.color, 3)
            text = "{:.2f}%".format(confidence) + ", " + str(self.im.calcDistance(x,y,w)) + "cm"
            cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)

        return self.people_detected
      
    def registerDetections(self, frame, gray, username, id):
        if len(self.faces) != 1:
            print("1 face should be detected by hc");
            return 1

        for (x,y,w,h) in self.faces:
            grayFace = gray[y:y+h,x:x+w]
            self.regFaces.append(grayFace)
            cv2.imwrite(self._temp_dir + username + "hc.%d.jpg"%(id),grayFace)

        return self.regFaces

class LBPDetector(BaseDetector):
    ###########################################################################
    # LBPDetector and HaarCascadeDetector class are exactly the same except for:
    #      self._classifier_file = self.path + GCo.LBP_CLASSIFIER_FILE ==> can be passed as parameter
    #     print("LBP Face Detection time=", end-start)  ==> can be passed as parameter too
    ###########################################################################
    def __init__(self):
        BaseDetector.__init__(self)
        self._classifier_file = self.path + GCo.LBP_CLASSIFIER_FILE
        self._face_cascade = None
        self.initialize()

    def isDetectorValid(self):
        return (self._face_cascade != None)

    def initialize(self):
        os.system("mkdir " + self._temp_dir)
        self.color = (255, 0, 0)
        if os.path.isfile(self._classifier_file):
            self._face_cascade = cv2.CascadeClassifier(self._classifier_file)

    def faceDetection(self, frame, gray):
        start = time.time()
        self.people_detected = 0
        self.audio = True
        self.faces = self._face_cascade.detectMultiScale(gray, 1.3, 5)
        end = time.time()
        print("LBP Face Detection time=", end-start)
        return self.faces

    def applyDetections(self, frame):
        self.people_detected = 0

        if self.faces == None:
            return 0

        for (x,y,w,h) in self.faces:
            confidence = 100
            self.people_detected += 1

            ###########################################
            # Save images to file
            ###########################################
            self.saveFaceToFile(frame, x, y, w, h)

            ###########################################
            # Draw the box and percentage around face
            ###########################################
            cv2.rectangle(frame, (x,y), (x+w,y+h), self.color, 3)
            text = "{:.2f}%".format(confidence) + ", " + str(self.im.calcDistance(x,y,w)) + "cm"
            cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)

        return self.people_detected
      
    def registerDetections(self, frame, gray, username, id):
        if len(self.faces) != 1:
            print("1 face should be detected by hc");
            return 1

        for (x,y,w,h) in self.faces:
            grayFace = gray[y:y+h,x:x+w]
            self.regFaces.append(grayFace)
            cv2.imwrite(self._temp_dir + username + "hc.%d.jpg"%(id),grayFace)

        return self.regFaces

class DnnDetector(BaseDetector):
    def __init__(self):
        BaseDetector.__init__(self)
        self._net = None
        self._prototxt = self.path + GCo.PROTOTXT_FILE
        self._model = self.path + GCo.CAFFEMODEL_FILE
        self.initialize()

    def isDetectorValid(self):
        return (self._net != None)

    def initialize(self):
        os.system("mkdir " + self._temp_dir)
        self.color = (0, 255, 255)
        if os.path.isfile(self._prototxt) and os.path.isfile(self._model):
            self._net = cv2.dnn.readNetFromCaffe(self._prototxt, self._model)

    def faceDetection(self, frame, gray):
        start = time.time()
        blob = cv2.dnn.blobFromImage(frame, 1.0, (GCo.GRIDX_SIZE, GCo.GRIDY_SIZE), (104.0, 177.0, 123.0))
        self._net.setInput(blob)
        dnnDetections = self._net.forward()
        lfaces = 0
        self.people_detected = 0
        self.faces = []

        ###########################################
        # Extact what current faces
        ###########################################
        if dnnDetections != None:
            self.audio = True
            for i in range(0, dnnDetections.shape[2]):
                ###########################################
                # Get confidence level and skip candidates with < 50%
                ###########################################
                confidence = dnnDetections[0, 0, i, 2]

                if confidence < GCo.CONFIDENCE_GRADE:
                    continue

                lfaces += 1

                ###########################################
                # calculate face box dimensions
                ###########################################
                box = dnnDetections[0, 0, i, 3:7] * np.array([GCo.DISPLAYX_SIZE, GCo.DISPLAYY_SIZE, GCo.DISPLAYX_SIZE, GCo.DISPLAYY_SIZE])
                (x1, y1, x2, y2) = box.astype("int")

                w = x2-x1
                h = y2-y1
                self.faces.append((x1+(w>>2),y1,w,h,confidence*100))

        self.people_detected = lfaces
        end = time.time()
        print("DNN Face Detection time=", end-start)
        return self.faces

    def applyDetections(self, frame):
        self.people_detected = 0

        if self.faces == None:
            return 0

        for (x,y,w,h,confidence) in self.faces:
            self.people_detected += 1

            ###########################################
            # Save images to file
            ###########################################
            self.saveFaceToFile(frame, x, y, w, h)

            ###########################################
            # Draw the box and percentage around face
            ###########################################
            cv2.rectangle(frame, (x,y), (x+w,y+h), self.color, 3)
            text = "{:.2f}%".format(confidence) + ", " + str(self.im.calcDistance(x,y,w)) + "cm"
            cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)

        return self.people_detected

    def registerDetections(self, frame, gray, username, id):
        if len(self.faces) != 1:
            print("1 face should be detected by dnn");
            return 1

        for (x,y,w,h,confidence) in self.faces:
            grayFace = gray[y:y+h,x:x+w]
            self.regFaces.append(grayFace)
            cv2.imwrite(self._temp_dir + username + "dnn.%d.jpg"%(id),grayFace)

        return self.regFaces

class FaceRecognizer():
    def __init__(self):
        self.db = DataBase()
        self.path = GCo.WORKING_DIRECTORY
        self.im = ImageManagement()
        self.LBPNames=[]
        self.EigenNames=[]
        self.FisherNames=[]
        self._audio = False
        self._LBPRecognizer = None
        self._EigenRecognizer = None
        self._FisherRecognizer = None
        self._LBPRecognizer_file = self.path + GCo.LBP_TRAINING_FILE
        self._EigenRecognizer_file = self.path + GCo.EIGEN_TRAINING_FILE
        self._FisherRecognizer_file = self.path + GCo.FISHER_TRAINING_FILE
        self.initialize()

    def isAudioEnabled(self):
        return self._audio

    def setAudioEnabled(self, enabled):
        self._audio = enabled

    def initialize(self):
        if GCo.LBP_ENABLED and os.path.isfile(self._LBPRecognizer_file):
            start = time.time()
            print("LBPHFaceRecognizer read starts")
            self._LBPRecognizer = cv2.face.LBPHFaceRecognizer_create()
            try: 
                self._LBPRecognizer.read(self._LBPRecognizer_file)
            except:
                print("Warning: lbp trainning file empty or already in use")
            print("LBPFaceRecognizer read done")
            end = time.time()
            print("LBP Loading Training file time=", end-start)

        if GCo.EIGEN_ENABLED and os.path.isfile(self._EigenRecognizer_file):
            start = time.time()
            print("EigenFaceRecognizer read starts")
            self._EigenRecognizer = cv2.face.EigenFaceRecognizer_create()
            try: 
                self._EigenRecognizer.read(self._EigenRecognizer_file)
            except:
                print("Warning: Eigen trainning file empty or already in use")
            print("EigenFaceRecognizer read done")
            end = time.time()
            print("EIGEN Loading Training file time=", end-start)

        if GCo.FISHER_ENABLED and os.path.isfile(self._FisherRecognizer_file):
            start = time.time()
            print("FisherFaceRecognizer")
            self._FisherRecognizer = cv2.face.FisherFaceRecognizer_create()
            try: 
                self._FisherRecognizer.read(self._FisherRecognizer_file)
            except:
                print("Warning: trainning file empty or already in use")
            print("FisherFaceRecognizer read done")
            end = time.time()
            print("FISHER Loading Training file time=", end-start)


    def refreshRecognizerData(self):
        self.initialize()
        print("Recognizer refreshed")

    def isRecognizerValid(self):
        return (self._LBPRecognizer != None)

    def getNames(self):
        return self.LBPNames, self.EigenNames, self.FisherNames

    def predict(self, frame, gray, x, y, w, h, det_conf):
        ###########################################
        # Extract face area and resize face it
        ###########################################
        try:
            faceImg = gray[y:y+h,x:x+w]
            faceImg = cv2.resize(faceImg,(GCo.GRIDX_SIZE, GCo.GRIDY_SIZE))
        except:
            print("Unable to process faces, aborting recognition")
            return

        ###########################################
        # Classify face by Local Binary Patterns Algoritm
        ###########################################
        try:
            if GCo.LBP_ENABLED:
                start = time.time()
                print("LPB starts")
                ids,conf = self._LBPRecognizer.predict(faceImg)
                print("LPB=", ids, conf)
                end = time.time()
                print("LBP Predict time=", end-start)

                result = self.db.getNamesFromIds(ids)
                if result == None:
                    print("Null result")
                    return result

                if len(result) == 0:
                    return "?",conf
                name = result[0][0]
                print(result, ids, conf)

                self.LBPNames.append({"name":name, "id":ids, "det_conf":det_conf, "conf": conf, "x":x, "y":y, "h":h})
        except:
            print("Exception occurred!")

        ###########################################
        # Classify face by Eigen Algoritm
        ###########################################
        try:
            if GCo.EIGEN_ENABLED:
                start = time.time()
                print("Eigen starts")
                ids,conf = self._EigenRecognizer.predict(faceImg)
                print("Eigen=", ids, conf)
                end = time.time()
                print("EIGEN Predict time=", end-start)

                result = self.db.getNamesFromIds(ids)
                if result == None:
                    print("Null result")
                    return result

                if len(result) == 0:
                    return "?",conf
                name = result[0][0]
                print(result, ids, conf)

                self.EigenNames.append({"name":name, "id":ids, "det_conf":det_conf, "conf": conf, "x":x, "y":y, "h":h})
        except:
            print("Exception occurred!")

        ###########################################
        # Classify face by Fisher algoritm
        ###########################################
        try:
            if GCo.FISHER_ENABLED:
                start = time.time()
                print("Fisher starts")
                ids,conf = self._FisherRecognizer.predict(faceImg)
                print("Fisher=", ids, conf)
                end = time.time()
                print("FISHER Predict time=", end-start)

                result = self.db.getNamesFromIds(ids)
                if result == None:
                    print("Null result")
                    return result

                if len(result) == 0:
                    return "?",conf
                name = result[0][0]
                print(result, ids, conf)

                self.FisherNames.append({"name":name, "id":ids, "det_conf":det_conf, "conf": conf, "x":x, "y":y, "h":h})
        except:
            print("Exception occurred!")

    def haarCascadeApplyRecognitions(self, frame, lpb_names, eigen_names, fisher_names):
        color = (0, 255, 0)

        if GCo.LBP_ENABLED:
            for person in self.LBPNames:
                 self.im.markFace(frame, self._audio,person["name"], person["id"], person["det_conf"],
                                  person["conf"], person["x"], person["y"], person["h"], color, GCo.LBP_RECOGNITION)

        if GCo.EIGEN_ENABLED:
            for person in self.EigenNames:
                 self.im.markFace(frame, False, person["name"], person["id"], person["det_conf"], 
                                  person["conf"], person["x"], person["y"], person["h"], color, GCo.EIGEN_RECOGNITION)

        if GCo.FISHER_ENABLED:
            for person in self.FisherNames:
                self.im.markFace(frame, False, person["name"], person["id"], person["det_conf"],
                person["conf"], person["x"], person["y"], person["h"], color, GCo.FISHER_RECOGNITION)

    def dnnApplyRecognitions(self, frame, lpb_names, eigen_names, fisher_names):
        color = (0, 255, 255)

        if GCo.LBP_ENABLED:
            for person in self.LBPNames:
                self.im.markFace(frame, self._audio,person["name"], person["id"], person["det_conf"],
                                 person["conf"], person["x"], person["y"], person["h"], color, GCo.LBP_RECOGNITION)

        if GCo.EIGEN_ENABLED:
            for person in self.EigenNames:
                self.im.markFace(frame, False, person["name"], person["id"], person["det_conf"],
                                 person["conf"], person["x"], person["y"], person["h"], color, GCo.EIGEN_RECOGNITION)

        if GCo.FISHER_ENABLED:
            for person in self.FisherNames:
                self.im.markFace(frame, False, person["name"], person["id"], person["det_conf"],
                                 person["conf"], person["x"], person["y"], person["h"], color, GCo.FISHER_RECOGNITION)

    def haarCascadeFaceRecognition(self, frame, detections, gray):
        self.LBPNames = []
        self.EigenNames = []
        self.FisherNames = []

        if detections == None:
            return self.LBPNames, self.EigenNames, self.FisherNames

        for (x,y,w,h) in detections:
            self.predict(frame, gray, x, y, w, h, 100)

        return self.LBPNames, self.EigenNames, self.FisherNames

    def lbpFaceRecognition(self, frame, detections, gray):
        return self.haarCascadeFaceRecognition(frame, detections, gray)

    def dnnFaceRecognition(self, frame, detections, gray):
        self.LBPNames = []
        self.EigenNames = []
        self.FisherNames = []

        if detections == None:
            return self.LBPNames, self.EigenNames, self.FisherNames

        for (x,y,w,h,confidence) in detections:
            self.predict(frame, gray, x, y, w, h, confidence)

        return self.LBPNames, self.EigenNames, self.FisherNames

class MainController():
    def __init__(self, trainer):
        self._trainer = trainer

        self._db = DataBase()
        self._path = GCo.WORKING_DIRECTORY
        self._camera = Camera()
        self._audioctl = AudioController()
        self._frame = None
        self._gray = None
        self._ret = 0
        self._dnnRegFaces = None
        self._hcRegFaces = None
        self._lbpRegFaces = None
        self._dnnFaces = None
        self._hcFaces = None
        self._lbpFaces = None
        self._triggerCounter = GCo.SKIP_FRAMES
        self._lbp_people_detected = 0
        self._hc_people_detected = 0
        self._dnn_people_detected = 0
        self.loadingError = False
        self.initialize()

    def initialize(self):
        self._nm = NavigationMenu()
        self._im = ImageManagement()
        self._nm.setForceDisplayEnabled(True)
        self._registration = Registration()

        print("instancing hr detector")
        self._hcDet = HaarCascadeDetector()
        if self._hcDet.isDetectorValid() == False:
            self.loadingError = True
            print("Error loading HaarCascade Detector")

        print("instancing lbp detector")
        self._lbpDet = LBPDetector()
        if self._lbpDet.isDetectorValid() == False:
            self.loadingError = True
            print("Error loading LBP Detector")

        print("instancing dnn detector")
        self._dnnDet = DnnDetector()
        if self._hcDet.isDetectorValid() == False:
            self.loadingError = True
            print("Error loading Dnn Detector")

        print("instancing hr recognizer")
        self._faceRec = FaceRecognizer()
        if self._faceRec.isRecognizerValid() == False:
            self.loadingError = True
            print("Error loading Face Recognizer")

    def takePicture(self, skipframes):
        if self._triggerCounter+1 >= GCo.SKIP_FRAMES and (self._nm.isDetectionEnabled() or self._nm.isRecognitionEnabled()):
            print("Forcing camera enabled")
            self._camera.setEnabled(True);

        if self._camera.isEnabled() or self._frame == None:
            for i in range(0,skipframes):
                ret, self._frame = self._camera.read()
            return ret, self._frame

        return 1,self._frame

    def displayImage(self):
        self._im.setStatus(self._frame, "0=quit,1=reg,3=cam on/off,4=dnn/haar/lbp,5=trig,7=det,9=rec", (100, 470))
        if self._nm.isDetectionEnabled() == False:
            if self._nm.isRegistrationMode():
                self._im.setHeader(self._frame, "In Registration mode", (10,30))
            elif self._camera.isEnabled():
                self._im.setHeader(self._frame, "detection disabled", (10,30))
            else:
                self._im.setHeader(self._frame, "detection disabled, camera disabled", (10,30))
        else:
            ###########################################
            # Add header Information
            ###########################################
            text = "capture on " + str(GCo.SKIP_FRAMES - self._triggerCounter)
            self._im.setHeader(self._frame, text, (10, 20))

            mode = "DETECTION"
            if self._nm.isRecognitionEnabled():
                mode = "RECOGNITION"

            method = "HAAR CASCADE"
            if self._nm.getDetectionMethod() == GCo.DNN:
                method = "DNN"
            elif self._nm.getDetectionMethod() == GCo.LBP:
                method = "LBP"

            self._im.setHeader(self._frame, method + "/" + mode, (250, 20))
            self._im.setStatus(self._frame, str(self._lbp_people_detected) + "/" + str(self._hc_people_detected) + "/" + str(self._dnn_people_detected), (10, 470))

        if self._camera.isEnabled() or self._nm.isForceDisplayEnabled():
            cv2.imshow(GCo.IMAGE_TITLE, self._frame)
            cv2.moveWindow(GCo.IMAGE_TITLE,10,10)

            if self._nm.isForceDisplayEnabled():
                self._nm.setForceDisplayEnabled(False)
                self._camera.setEnabled(False)

            if self._triggerCounter == 0 and (self._nm.isDetectionEnabled() or self._nm.isRecognitionEnabled()):
                self._im.writeImage(self._frame, path=self._path + "detections/", play_sound=False)

        return self._nm.processKey(self._frame, 50, 0)

    def captureFace(self):
        self._gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)

        username = self._registration.getUserName()
        print (username)
        face_detected = False

        if self._lbpDet.getRegistrationsNumber() < GCo.MAX_REG_PICTURES:
            print("lbp next registration")
            self._lbpRegFaces = self._lbpDet.faceDetection(self._frame, self._gray)
            lbpExtractedFaces = []
            if len(self._lbpRegFaces) > 0:
                face_detected = True
                self._audioctl.cmdSound(GCo.AU_FACEDETECTEDLBP, wait=True)
                id = self._registration.getNextCounter()
                lbpExtractedFaces = self._lbpDet.registerDetections(self._frame, self._gray, username, id)

        if self._hcDet.getRegistrationsNumber() < GCo.MAX_REG_PICTURES:
            print("hc next registration")
            self._hcRegFaces = self._hcDet.faceDetection(self._frame, self._gray)
            hcExtractedFaces = []
            if len(self._hcRegFaces) > 0:
                face_detected = True
                self._audioctl.cmdSound(GCo.AU_FACEDETECTEDHC, wait=True)
                id = self._registration.getNextCounter()
                hcExtractedFaces = self._hcDet.registerDetections(self._frame, self._gray, username, id)

        if self._dnnDet.getRegistrationsNumber() <  GCo.MAX_REG_PICTURES:
            print("dnn next registration")
            self._dnnRegFaces = self._dnnDet.faceDetection(self._frame, self._gray)
            dnnExtractedFaces = []
            if len(self._dnnRegFaces) > 0:
                face_detected = True
                self._audioctl.cmdSound(GCo.AU_FACEDETECTEDDNN, wait=True)
                id = self._registration.getNextCounter()
                dnnExtractedFaces = self._dnnDet.registerDetections(self._frame, self._gray, username, id)

        if face_detected == False:
            self._audioctl.cmdSound(GCo.AU_NOFACES, wait=True)

        if self._lbpDet.getRegistrationsNumber() >= GCo.MAX_REG_PICTURES and\
           self._hcDet.getRegistrationsNumber() >= GCo.MAX_REG_PICTURES and\
           self._dnnDet.getRegistrationsNumber() >= GCo.MAX_REG_PICTURES:
            print("Registration completed!")
            self._registration.stopRegistration()
            self._nm.setRegistrationMode(False)
            self._camera.setEnabled(False)
            self._trainer.addUserToDB(username, id)
            self._trainer.trainAll()
            self._faceRec.refreshRecognizerData()

#        time.sleep(0.01)

        text = "LBP=" + str(self._lbpDet.getRegistrationsNumber())+"/"+str(GCo.MAX_REG_PICTURES)
        text += "HC=" + str(self._hcDet.getRegistrationsNumber())+"/"+str(GCo.MAX_REG_PICTURES)
        text += ", DNN=" + str(self._dnnDet.getRegistrationsNumber())+"/"+str(GCo.MAX_REG_PICTURES)
        cv2.putText(self._frame, text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        rc = self.displayImage()
        if rc == GCo.NM_QUIT:
           if self._nm.isRegistrationMode():
               self._audioctl.cmdSound(GCo.AU_REGISTRATIONABORTED, wait=True)
               self._nm.setRegistrationMode(False)
               self._camera.setEnabled(False)
               return 0
           else:
               self._audioctl.cmdSound(GCo.AU_QUIT, wait=True)
               return 1
        return 0

    def executeDetection(self):
        self._triggerCounter += 1
        lfaces = 0
        if self._nm.isTriggerFlagEnabled():
            self._triggerCounter = GCo.SKIP_FRAMES
            self._nm.setTriggerFlagEnabled(False)

        if self._triggerCounter >= GCo.SKIP_FRAMES:
            self._triggerCounter = 0
                
            self._gray = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)

            ###########################################
            # Detect faces 
            ###########################################
            print("Face Detection started...")
            people_detected = 0
            self._lbp_people_detected = 0
            self._hc_people_detected = 0
            self._dnn_people_detected = 0

            if self._nm.getDetectionMethod() == GCo.HAAR_CASCADE:
                self._hcFaces = self._hcDet.faceDetection(self._frame, self._gray)
                people_detected = len(self._hcFaces)
                self._hc_people_detected = people_detected
            elif self._nm.getDetectionMethod() == GCo.DNN:
                self._dnnFaces = self._dnnDet.faceDetection(self._frame, self._gray)
                people_detected = len(self._dnnFaces)
                self._dnn_people_detected = people_detected
            else: # LBP
                self._lbpFaces = self._lbpDet.faceDetection(self._frame, self._gray)
                people_detected = len(self._lbpFaces)
                self._lbp_people_detected = people_detected

            self._camera.setEnabled(False)
            self._nm.setForceDisplayEnabled(True)
            self._audioctl.cmdSound(GCo.AU_DETECTIONDONE, wait=True)
            self._audioctl.cmdPeopleDetected(people_detected, wait=True)
            print("Face Detection done. " + str(people_detected) + " people detected")
            return people_detected

        return 0

    def executeRecognition(self):
	print("Face Recognition started...")
	if self._nm.getDetectionMethod() == GCo.HAAR_CASCADE:
            self._faceRec.setAudioEnabled(True)
	    self._faceRec.haarCascadeFaceRecognition(self._frame, self._hcFaces, self._gray)
	elif self._nm.getDetectionMethod() == GCo.DNN:
            self._faceRec.setAudioEnabled(True)
	    self._faceRec.dnnFaceRecognition(self._frame, self._dnnFaces, self._gray)
	else: #LBP
            self._faceRec.setAudioEnabled(True)
	    self._faceRec.lbpFaceRecognition(self._frame, self._lbpFaces, self._gray)

	print("Face Recognition done.")
	self._audioctl.cmdSound("recognitiondone", wait=True)
	self._audioctl.cmdSound("peoplerecognized", wait=True)


    def applyDetectionsAndDisplay(self):
        ##############################################################################
        # If time to process a new image
        ##############################################################################
        if self._triggerCounter == 0 and (self._nm.isDetectionEnabled() or self._nm.isRecognitionEnabled()):
            filename = self._im.writeImage(self._frame, self._path + GCo.IMAGES_BEF_PROCESS, play_sound=False)
            print(filename)

            if self._nm.getDetectionMethod() == GCo.LBP and self._lbpFaces != None and len(self._lbpFaces) > 0:
                self._lbpDet.applyDetections(self._frame)
                lbp_names, eigen_names, fisher_names = self._faceRec.getNames()
                if len(lbp_names) > 0 or len(eigen_names) > 0 or len(fisher_names) > 0:
                    self._faceRec.haarCascadeApplyRecognitions(self._frame, lbp_names, eigen_names, fisher_names)
                    self._faceRec.setAudioEnabled(False)
            elif self._nm.getDetectionMethod() == GCo.HAAR_CASCADE and self._hcFaces != None and len(self._hcFaces) > 0:
                self._hcDet.applyDetections(self._frame)
                lbp_names, eigen_names, fisher_names = self._faceRec.getNames()
                if len(lbp_names) > 0 or len(eigen_names) > 0 or len(fisher_names) > 0:
                    self._faceRec.haarCascadeApplyRecognitions(self._frame, lbp_names, eigen_names, fisher_names)
                    self._faceRec.setAudioEnabled(False)
            elif self._nm.getDetectionMethod() == GCo.DNN and self._dnnFaces != None and self._dnnFaces > 0:
                self._dnnDet.applyDetections(self._frame)
                lbp_names, eigen_names, fisher_names = self._faceRec.getNames()
                if len(lbp_names) > 0 or len(eigen_names) > 0 or len(fisher_names) > 0:
                    self._faceRec.dnnApplyRecognitions(self._frame, lbp_names, eigen_names, fisher_names)

        ##############################################################################
        # Display Image
        ##############################################################################
        rc = self.displayImage()

        ##############################################################################
        # If Start Registration
        ##############################################################################
        if rc == GCo.NM_TRAINNOW:
            print("Registration to be started")
            self._dnnRegFaces = []
            self._hcRegFaces = []
            self._lbpRegFaces = []
            self._lbpDet.clearRegistrations()
            self._hcDet.clearRegistrations()
            self._dnnDet.clearRegistrations()
            self._camera.setEnabled(True)
            username = self._registration.startRegistration(self._frame)
            if username != "":
                self._nm.setRegistrationMode(True)
        ##############################################################################
        # If Requested to write an image
        ##############################################################################
        elif rc == GCo.NM_WRITEIMAGE:
            self._camera.setEnabled(False)
            filename = self._im.writeImage(self._frame, self._path)
            print("File saved into",filename)
        ##############################################################################
        # If Asking to quit application
        ##############################################################################
        elif rc == GCo.NM_QUIT:
            self._audioctl.cmdSound(GCo.AU_LEAVING, wait=True)
            while True:
                cv2.imshow(GCo.IMAGE_TITLE, self._frame)
                key = cv2.waitKey(50)
                kpkey = Keypad().readKey()
                if key != -1 or kpkey != -1:
                    break;

            if key == ord("a") or key == ord("1") or key == 177 or kpkey == 1:
               self._audioctl.cmdSound(GCo.AU_QUIT, wait=True)
               return 1
            elif key == ord("i") or key == ord("9") or key == 185 or kpkey == 9:
               os.system("sudo shutdown 0")
            elif key == ord("e") or key == ord("5") or key == 181 or kpkey == 0:
               os.system("reboot")

        return 0

    def run(self):
        #if self.loadingError:
        #    return 1

        # Skip N frames to give time raspberry pi to be better responsive
        skipFrames = 3
        if self._nm.isRegistrationMode():
            self._audioctl.cmdSound(GCo.AU_PHOTONOW, wait=True)
            skipFrames = 60

        self._ret, xframe = self.takePicture(skipFrames)
        if self._ret == 0:
           self._frame = xframe

        self._gray = None

        if self._nm.isRegistrationMode():
           return self.captureFace()
        elif self._nm.isDetectionEnabled():
           people_detected = self.executeDetection()
           if self._nm.isRecognitionEnabled() and people_detected > 0:
               self.executeRecognition()
 
        return self.applyDetectionsAndDisplay()


def getDetectionsDir(path):
    return path + '{:%Y-%m-%d}'.format(datetime.datetime.now())    


###########################################
# Main Entry point
###########################################
if __name__ == '__main__':
    audioctl = AudioController()
    audioctl.cmdSound(GCo.AU_WELCOME, wait=True)
    audioctl.cmdSound(GCo.AU_LOADING, wait=False)

    db = DataBase()

    trainer = Trainer()

    # NOTE:
    #    Enable this condition if willing to rebuild the whole db
    #    and retrain database during loading
    if False:
        trainer.trainAll()

    camera = Camera()
    camera.start()

    controller = MainController(trainer)

    audioctl.cmdSound(GCo.AU_READY, wait=False)

    ###########################################
    # Enter to a loop reading for camera frames
    ###########################################
    while True:
        if controller.run():
            print("Closing controller")
            break;
    
    camera.close()

    Keypad().exit()
    GPIO.cleanup()
