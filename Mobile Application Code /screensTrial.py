import wave, threading, string,tempfile, pyaudio, audioop, socket, kivy, kivymd, os, time, datetime as dt, sys, matplotlib, cv2, requests, hashlib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import boto3, librosa, matplotlib.pyplot as plt, numpy as np
from sklearn.linear_model import LogisticRegression
import json
from io import BytesIO
import matplotlib.pyplot as plt
import librosa, librosa.display
#builder that allows loading any kv app no matter the name
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup
from kivymd.uix.textfield import MDTextField, MDTextFieldRect
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.graphics import Rectangle, Color, Line, RoundedRectangle, Ellipse
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRoundFlatIconButton, MDRoundFlatButton, MDFloatingActionButton, MDFlatButton, MDIconButton, MDFillRoundFlatButton
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.bottomnavigation import MDBottomNavigation,MDBottomNavigationItem
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.dialog import MDDialog
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from kivy.core.audio import SoundLoader
import sounddevice as sd
import soundfile as sf
from kivy.clock import Clock
from datetime import date, datetime
import numpy as np
from kivy.core.audio import Sound
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from kivymd.uix.relativelayout import RelativeLayout
from kivymd.uix.card import MDCard
from kivy.properties import ObjectProperty, ListProperty, DictProperty
# from kivy.garden.matplotlib import FigureCanvasKivyAgg
# from kivy_garden.graph import Graph, MeshLinePlot
# from validate_email import validate_email
from email_validator import validate_email, EmailNotValidError
# from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.metrics import dp, sp
# import screeninfo
import multiprocessing
from kivy.config import Config
from kivymd.app import MDApp
from keras.models import load_model
import base64


# light sql file on the device

#Window.fullscreen = 'auto'
#Config.set('graphics', 'width', '420')
#Config.set('graphics', 'height', '800')
# Config.set('kivy', 'keyboard_mode', 'systemanddock')
Config.set('graphics', 'resizable', True)
# width = screeninfo.get_monitors()[0].width
# height = screeninfo.get_monitors()[0].height 

width=380
height=800

class BackgroundColor(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(1,1,1,1) 
            Rectangle(size=(width,height))
class BackgroundBox(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(134/256, 162/256, 182/256,9)
            RoundedRectangle(size=(width*0.9,height*0.95),radius=[30,30,30,30],pos=(width*0.05,height*0.025))
class cornerBox(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(205/256, 232/256, 252/256,1)
            RoundedRectangle(size=(width*0.7,height*0.3), radius=[0,0,0,320], pos=(width*0.3,height*0.82),start_angle=90, end_angle=360)
class bottomCornerBox(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(212/255,233/255,250/255,1)
            RoundedRectangle(size=(width+100,height*0.7), radius=[260,0,0,0], pos=(width*0.3,0))
class mainBox(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(175/256, 202/256, 222/256,1)
            RoundedRectangle(size=(width,height*0.6), radius=[100,240,0,200])
class slBoxes(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(205/256, 232/256, 252/256,0.6)
            RoundedRectangle(size=(width+100,height), radius=[0,320,0,0], pos=(-100,0))
            Color(155/256, 182/256, 202/256,0.8)
            RoundedRectangle(size=(width,height*0.72), radius=[240,100,200,0])
class homeBox(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(155/256, 182/256, 202/256,0.9)
            RoundedRectangle(size=(width*0.9,height*0.52), radius=[30,80,30,80], pos=(width*0.05,height*0.05), elevation=2)
class bottomCornerBox2(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(205/256, 232/256, 252/256,1)
            RoundedRectangle(size=(450,height*0.55), radius=[260,0,0,0], pos=(width*0.2,0))
class fpWindow(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(205/256, 232/256, 252/256,1)
            RoundedRectangle(size=(width*2,height*0.9), radius=[360,0,0,0], pos=(0,0))
            Color(155/256, 182/256, 202/256,0.95)
            RoundedRectangle(size=(width*0.95,height*0.67), radius=[100,200,100,200], pos=(width*0.025,height*0.15))
class emotionCircle(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(155/256, 182/256, 202/256,0.9)
            Ellipse(size=(width*0.9,width*0.9), pos=(width*0.05,height*0.35))

class Drawer(MDBoxLayout):
    pass
class noWifi(MDDialog):
    pass
# class noConnectionPopup(FloatLayout):
#     pass
no_connection=None
def is_connected():
    try:
        socket.create_connection(("1.1.1.1", 53))
        return True
    except OSError:
        pass
    return False
def show_popup():
    no_connection=MDDialog(title='Wifi Connection', 
                           text='Please connect to the wifi in order to proceed',
                        #    size_hint=[height*0.3,width*0.8],
                        #    pos_hint={"y":0.35,"x":0.1}
                           buttons=[MDFlatButton(text='OK')])
                                                 #,on_release=self.check_connection())])
    no_connection.open()
def check_connection():
    if is_connected()==True:
        show_popup()
    else:
        no_connection.dismiss()
        
#Class for every window
class IntroWindow(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(BackgroundColor())
        self.add_widget(cornerBox())
        self.add_widget(mainBox())



import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate({
  "type": "service_account",
  "project_id": "access-elai",
  "private_key_id": "2bc85f08135cf6bac17889641d94616359854e08",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDbA8hGhu4r/Jg+\nEh3RVFmNJUdJFyeWgivvq/9nmXLT5AwRgrSl7APHa50+duoyqY3SKMicjRclIwBk\nrGf2FizRUJJh7NiFf9LIyE5Aa3TrsMlH8JW2AiQZc9qOdRV3oI5XlBOTuYelDYQF\nsZdnhrS2/Jhz7oROopBTQXK/u2+UoSosCj+puj400fe7I9Jj+OsMqIhHpm+h9Bok\n7PGB1lE6KrbX99lVUreV/zlM/PVRGhgtfiaGGALKhZKI+ww2JsKdOXiWLHnfmZzV\nPn3htRNdLJFb4KRMLz3GECGtf+PukBy6AHRMdK3LZRUi0keuLugph/SsdH9/IPOA\n6t+sXU+tAgMBAAECggEACs8oesadCaG5V9LoEtVBaDRvTL2uUADTr0wDWncZ0jhe\nmhyj0s6Pry9x/su6qk5w9+7YW9Wgz03nbpCNvvkANEJwPxID667P1eYA6rADAMDk\nZj8K/IUlh/YOUtqXeSR88fiWcMOG3NAKdId7y/m7gI597bbXY9QlIF7KDYlwK9sj\nCUtj8HPBk59G49HyX8lgtqI80h6jeuwW+Wf/VajlfpdSwqtWIagIPaO7h4OTE4ES\nggwFRO3+fTv5uateOGg9LogCw9ib7XWmwk3P8pa5mmWQyLmjehvAPCIWCc0aDbWX\nGUqrfY846ph/wQYs12xB1aJRU05Ah1V9nDhQRVneAQKBgQD07ln4psyCiPunTf9+\nk6818PfRx7bemcEosmXWntqk3TrLqHhynKHTT0XFG6jLLtnAVvJER1jFwF/M2p9q\nwkOOagn2G5rMb61CGeZ3yp+zdQYo1qQJyQDFCvvgzkfaeSOZcwk5XhTMtgmrIc1o\nDdRSvE1sHA6EsJdYg2u3Iz1arQKBgQDk6ZncQUfg/6AjJZEPQQ1RimufFWhWJF6j\n7Gp5dFt32aWw2uHWAJUy0NZof8FC3GSDHzNpjOIwKQR+hEi5LMcFSLzhvj/bkwTA\n/i12aDwaDNxaLAFEYZr0XfopN0xPXDRLVHBhp8JCr4qjESGJCDngMHi01j07dnw7\n6xnkD4tpAQKBgQDD2lhdaVt9QfYhSVB1MbjYJFC2EcHb7Ay18zlVzf69+B1MvvFZ\njIAmTWxX+g8WMedzUtM03+xPbM3uLB9vqdmFZquCfX5h3ScpBTbyMTdUs83yF/hh\nzrXr2iWhFLIGM/nQeVk141I0g5flnQj4HJ7cbbBnM2Q0nFTZNXWLowUrnQKBgAqx\n92RZEHisuNiriqmBypOCuiCGqYdMz7cs9pSSISvqWVl4AJE1GcN0CnB7d5YeIfwW\nWxqVYIQLhpA6sgMk2m+exGRvtSAXMGOr/IfJuvUkoK7921lMjibYtTVzxfb3QeI7\nIb0OT386IGoaBM0YO0wEN7+LOvUqRgeuplkHeOYBAoGATKdHesnQCWElxyJbVLY+\nJnLB6bVxxammUvwiSa5nNsfKJ97C8KaGEF0rwIZNOzcCG8RsYsMSECV0Da8ccOvu\nMcjs3bZ0QGQPq7au8dc4RDUrBb0Z0u0nenMHN6iRFIHdW8zpvwNLYfiicX2KnuNN\n6K7BRyO02C+aPyvNNQcukSI=\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-m6vvj@access-elai.iam.gserviceaccount.com",
  "client_id": "114479693391035249571",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-m6vvj%40access-elai.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
})

firebase_admin.initialize_app(cred, {'databaseURL': 'https://access-elai-default-rtdb.firebaseio.com/'})

creds = []

for i in range(1,6):
    ref = db.reference('/').child(str(i))
    data = ref.get()
    creds.append(data)

session = boto3.Session(aws_access_key_id=creds[2],aws_secret_access_key=creds[3],aws_session_token=creds[4],region_name=creds[0])

class createWindow(Screen):
    email, fullname, username, day, month, year, passW, confirmed= ObjectProperty(str),ObjectProperty(str),ObjectProperty(str),ObjectProperty(int),ObjectProperty(int),ObjectProperty(int),ObjectProperty(str),ObjectProperty(str)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(BackgroundColor())
        self.add_widget(slBoxes())
        # sm.add_widget(createWindow(name='signup'))
        # print(self.ids)

    def validate_username(self):
        self.dynamodb = session.client('dynamodb')
        params = {'TableName': 'user',
                    'Key': {'userName': {'S': self.username.text}}}
        self.response = self.dynamodb.get_item(**params)
        try:
            self.user = self.response['Item']['user']['S']
        except:
            self.user = 'unknown'
        if self.user != 'unknown':
            self.username.helper_text= 'Username already taken'
            self.username.error=True
            return False
        return True
    
    def validateEmail(self):
        try:
            if validate_email(self.email.text):
               return True
        except EmailNotValidError:
            self.email.helper_text='Please enter a valid email address'
            self.email.error=True
            return False
        
    def validPassword(self):
        if len(self.passW.text)>=8 and len(set(self.passW.text)&set(string.ascii_lowercase))>0 and len(set(self.passW.text)&set(string.ascii_uppercase))>0 and len(set(self.passW.text)&set(string.digits))>0:
            #and len(list(set(self.passW.text)&set(string.punctuation)))>=0
            return True
        else: 
            self.passW.helper_text='Weak Password'
            self.passW.error=True
            return False
        
    def match(self):
        if self.passW.text!=self.confirmed.text: 
            self.confirmed.helper_text= "Passwords don't match"
            self.confirmed.error=True
            return False
        return True
    
    def get_location(self):
        ip_address = requests.get('https://api.ipify.org').text
        url = f'https://ipapi.co/{ip_address}/json/'
        response = requests.get(url)
        data = response.json()
        return [data.get('city', ''),data.get('country', '')]
    
    def validate(self):
        if self.validate_username() & self.validateEmail() & self.validPassword() & self.match():
            if len(self.email.text)>0 and len(self.fullname.text)>=4 and len(self.username.text)>=6 and len(self.day.text)==2 and len(self.month.text)==2 and len(self.year.text)==4:
                current=self.day.text+'/'+self.month.text+'/'+self.year.text
                DOB=datetime.strptime(current,'%d/%m/%Y')
                if DOB>=datetime.strptime('01/01/1920','%d/%m/%Y') and DOB<datetime.now():
                    dynamodb = session.client('dynamodb')
                    params = {
                        'TableName': 'users',
                        'ProjectionExpression': 'userName',
                        'Limit': 1,
                    }
                    now= str(datetime.now())
                    response = dynamodb.scan(**params)
                    items = response.get('Items', [])
                    # max_value = max(items, key=lambda x: x['userName'])['userName']
                    # global user_id
                    # user_id = str(int(max_value['N'])+1)
                    gfg = hashlib.sha3_256()
                    pas = self.passW.text+now
                    gfg.update(pas.encode(encoding = 'UTF-8'))
                    passw = gfg.digest()
                    loc = self.get_location()
                    info = {'city': {'S': loc[0] }, 
                            'country': {'S': loc[1]}, 
                            'dateOfBirth': {'S': str(DOB)}, 
                            'email': {'S': self.email.text}, 
                            'fullName': {'S': self.fullname.text},
                            'password': {'S': str(passw) }, 
                            'registrationDate&Time': {'S': now}, 
                            'userName': {'S': self.username.text}}
                    dynamodb.put_item(TableName='user', Item=info)
                    return True
        return False
    
    def showPassw(self):
        if self.passW.password == True:
            self.passW.password = False

    def hidePassw(self):
        if self.passW.password == False:
            self.passW.password = True

    def showPassword(self):
        if self.confirmed.password == True:
            self.confirmed.password = False

    def hidePassword(self):
        if self.confirmed.password == False:
            self.confirmed.password = True

class loginWindow(Screen):
    userInput, passInput=ObjectProperty(str),ObjectProperty(str)

    username = ''
    # info = DictProperty()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(BackgroundColor())
        self.add_widget(slBoxes())
        # sm.add_widget(loginWindow(name='login'))
        # print(sm.screens)
        # sm=app.root.ids.manager

    def confirm_user(self):
        self.dynamodb = session.client('dynamodb')
        params = {'TableName': 'user',
                    'Key': {'userName': {'S': self.userInput.text}}}
        self.response = self.dynamodb.get_item(**params)
        try:
            self.username = self.response['Item']['userName']['S']
        except:
            self.username = 'unknown'
        if self.username == 'unknown':
            self.userInput.helper_text= 'Username does not exist'
            self.userInput.error=True
            return False
        loginWindow.username = str(self.userInput.text)
        return True

    def showPassword(self):
        if self.passInput.password == True:
            self.passInput.password = False

    def hidePassword(self):
        if self.passInput.password == False:
            self.passInput.password = True

    def clearInputs(self):
        self.ids.userInput.text,self.ids.passInput.text='',''

    def confirm_login(self):
        if self.confirm_user():
            date_value = self.response['Item'].get('registrationDate&Time', {}).get('S', None)
            pass_value = self.response['Item'].get('password', {}).get('S', None)
            gfg = hashlib.sha3_256()
            pas = self.passInput.text+str(date_value)
            gfg.update(pas.encode(encoding = 'UTF-8'))
            passw = gfg.digest()
            if str(passw)==str(pass_value):
                return True
            else:
                self.passInput.helper_text='Incorrect password'
                self.passInput.error=True
                return False
        return False
    
    # def set_user(self):
    # # create grid to fix it
    #     home, account, edit=homeWindow(), accountWindow(), editProfile()
    #     # home.ids.users_username.text=self.username
    #     home.set_username()
    #     account.update_user_info()
    #     edit.update_user_profile()
    #     return True

# class soundWave(FigureCanvasKivyAgg):
#     def __init__(self, **kwargs):
#         super(soundWave, self).__init__(plt.gcf(), **kwargs)

class homeWindow(Screen):
    FM=None
    status= ObjectProperty(str)
    user = loginWindow.username
    rec_time = ''
    file_name = ''
    date = ''
    time = ''
    emotion = 3
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(BackgroundColor())
        self.add_widget(bottomCornerBox2())
        self.add_widget(homeBox())
        # self.add_widget(soundWave())
        self.path=os.path.expanduser("~") or os.path.expanduser("/")
        self.FM= MDFileManager(
            select_path=self.select_path,
            exit_manager=self.close_fm,
            background_color_toolbar=(155/256, 182/256, 202/256,1),
            md_bg_color=(0.97, 0.98, 0.99, 1),
            icon_color=(155/256, 182/256, 202/256,1),
            background_color_selection_button=(155/256, 182/256, 202/256,1),
            ext=[".mp3", ".pcm", ".wav", ".aac", ".wma", ".m4a"])
            #,search='files')
    
    def trigger(self):
        audio = self.preprocess_audio(self.recording)
        self.add_processed_to_bucket(audio)
        # self.predict(self.array)
        self.lambda_function()
        self.add_to_database()
        # self.add_recording_to_bucket()
        # print(self.audio.shape, 'trigger')
        # self.pipeline = Pipeline([('preprocess', FunctionTransformer(self.preprocess_audio)),
        #                           ('store_processed', FunctionTransformer(self.add_processed_to_bucket)),  
        #                           ('classify', FunctionTransformer(self.lambda_function)),
        #                           ('database', FunctionTransformer(self.add_to_database())),
        #                           ('store_recording', FunctionTransformer(self.add_recording_to_bucket))#,
                                #   ('dynamo', self.add_to_database(self.info))
                                #   ])#,
        # ('lambda', self.lambda_function())])
        # self.emotion = self.pipeline.transform(self.recording)
        #self.emotion=
        # self.pipeline.transform(self.audio)
        # self.emotion = 0

    def add_processed_to_bucket(self, rec):
        rec = np.array(rec).tobytes()
        s3_client = session.client('s3')
        bucket_name = 'elai-processed-recordings'
        # buffer = BytesIO(img)
        # buffer = BytesIO(rec)
        # rec_bytes = rec if isinstance(rec, bytes) else rec.encode()
        s3_client.put_object(Body=rec, Bucket= bucket_name, Key=self.file_name)

    # def add_recording_to_bucket(self):
        # print(self.file_name, 'addrtob')
        # s3_client = session.client('s3')
        # bucket_name = 'elai-user-recordings'
        # # buffer = BytesIO(img)
        # s3_client.put_object(Body=self.recording.tobytes(), Bucket= bucket_name, Key=self.file_name)
        # return

    def add_to_database(self):
        # return
        dynamodb_client = session.client('dynamodb')
        table_name = 'user-recordings'
        key = {
            'recordingID': {'S': self.file_name} #,
            # 'your_sort_key': {'N': 'your_sort_key_value'}
            }
        record = {'recordingID': {'S': self.file_name}, 
                            'date': {'S': self.date}, 
                            'time': {'S': self.time}, 
                            'duration': {'S': '5'},
                            'emotion': {'S': self.emotion}
                }
        dynamodb_client.put_item(TableName=table_name, Item=record)
        # response = dynamodb_client.get_item(TableName=table_name,Key=key)
        # print(response['Item'])
        # userDict = response['Item']
        # print(list[userDict.values()])

    # def preprocess_audio(self, audio):
    #     data=[]
    #     audio = np.array(self.recording)
    #     audio = audio.reshape(1, -1) 
    #     mfcc = librosa.feature.mfcc(y=audio, sr=44100 , n_mfcc=13)
    #     mfcc = np.mean(mfcc, axis=0)
    #     spectrogram = librosa.feature.melspectrogram(y=audio, sr=44100 , n_mels=128, fmax=8000) 
    #     log_spectrogram = np.mean(librosa.power_to_db(spectrogram), axis=0)
    #     chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=44100), axis=0)
    #     zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio), axis=0)
    #     rmse = np.mean(librosa.feature.rms(y=audio), axis=0)
    #     data = [mfcc, log_spectrogram, chroma, zcr, rmse]
    #     array = np.array(data)
    #     array = np.transpose(array) 
    #     self.array = [np.mean(array[i], axis=0).tolist() for i in range(len(array))]
        # self.array = np.array(self.array).reshape([1,397,5])
        # array= np.array(self.array).tobytes()
        # print(bytes_data)
        # byte_list = [bytes(x) for x in array]
        # bytes_data =  b''.join(byte_list)        
        # return self.array
    
    # def predict(self, processed):
    #     s3 = session.client('s3')
    #     m_Bucket = 'elai-processed-recordings'
    #     m_Name = 'model-1DCNN.h5'
    #     m = s3.get_object(Bucket=m_Bucket, Key=m_Name)['Body'].read()
    #     buffer = BytesIO(m)
    #     with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
    #         temp_file.write(buffer.getvalue())
    #         temp_file_path = temp_file.name
    #     model = load_model(temp_file_path, compile=False)
    #     temp_file.close()
    #     prediction = np.argmax(model.predict(processed))
    #     return prediction

    def lambda_function(self):
        # event = {}
        # event.update({"one": self.array[0]})
        # event.update({"two": self.array[1]})
        # event.update({"three": self.array[2]})
        # event.update({"four": self.array[3]})
        # event.update({"five": self.array[4]})

        # json_str = json.dumps(event)
        # bytes_data = json_str.encode('utf-8')
        # encrypted_data = base64.b64encode(bytes_data).decode('utf-8')
        # print(encrypted_data)
        # event = {"filename": str(self.file_name), "payload": encrypted_data}
    
        # encrypted_data = base64.b64encode(event).decode("utf-8")
        # event={"filename": str(self.file_name)}
        event = {}
        event.update({"one": self.array[0]})
        event.update({"two": self.array[1]})
        event.update({"three": self.array[2]})
        event.update({"four": self.array[3]})
        event.update({"five": self.array[4]})
        json_str = json.dumps(event)
        bytes_data = json_str.encode('utf-8')
        encrypted_data = base64.b64encode(bytes_data).decode('utf-8')

        lambda_client = session.client('lambda')
        response = lambda_client.invoke(
            FunctionName='classification',
            Payload=json.dumps({"data":encrypted_data}),
        )
        if response['StatusCode'] == 200:
            lambda_response = json.loads(response['Payload'].read().decode('utf-8'))
            print(lambda_response['body'])
            try: 
                dic = json.loads(lambda_response['body'])
                val = dic.keys()[-1]
                # print(lambda_response['body']['prediction class'])
                self.emotion= dic[val]
                print("emotion from lambda")
            except:
                self.emotion=0
            # self.emotion = int(json.loads(lambda_response['body'])['prediction class'])

    def open_fm(self):
        self.FM.show(self.path)

    def select_path(self, path:str):
        print(path)
        self.audio = path
        self.classify_upload()
        self.close_fm()
    
    def close_fm(self, *args):
        self.FM.close()

    def remove_widgets(self):
        try:
            self.ids.grid_emotion.remove_widget(self.emotionOutput)
            self.ids.grid_emoticon.remove_widget(self.emotionButton)
        except:
            pass

    def start_recording(self):
        # self.remove_widgets()
        sample_rate = 22050
        duration= 5
        x=datetime.now()
        self.user = loginWindow.username
        self.rec_time, self.date, self.time = str(x.strftime("%d%m%y%H%M%S")),str(x.strftime("%d/%m/%y")),str(x.strftime("%H:%M:%S")) 
        key_name = str(str(self.user) + self.rec_time)
        self.file_name = key_name
        s3_client = session.client('s3')
        bucket_name = 'elai-user-recordings'
        recording_buffer = BytesIO()
        self.recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        sf.write(recording_buffer, self.recording, sample_rate, format='wav')
        recording_buffer.seek(0)
        s3_client.put_object(Body=recording_buffer, Bucket=bucket_name, Key=self.file_name)
        self.recording = self.recording[17640:]
        self.trigger()
        # self.emotion_list=['Neutral','Happy','Sad','Anger','Fear','Disgust']
        # self.emoticons=['emoticon-happy-outline','emoticon-excited-outline','emoticon-sad-outline','emoticon-angry-outline','emoticon-frown-outline','emoticon-sick/neutral-outline']
        # self.emotion_icon = self.emoticons[self.emotion]
        # self.emotionButton=MDRoundFlatIconButton(icon=self.emotion_icon, size_hint=(0.8,width*0.8), md_bg_color=(0,0,0,0), line_color=(0,0,0,0), icon_color=(0.1,0.1,0.2,1), icon_size='240sp')
        # self.emotionOutput=MDRoundFlatIconButton(text=self.emotion_list[self.emotion],font_size='44sp',size_hint_x=(0.8), md_bg_color=(0,0,0,0),  line_color=(0,0,0,0), text_color=(0.1,0.1,0.2,1), halign='center')
        # self.ids.grid_emoticon.add_widget(self.emotionButton)
        # self.ids.grid_emotion.add_widget(self.emotionOutput)
        # print(self.emotion)
        # emotion = self.emotion
        # self.emotion_list=["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad","Surprise","Calm"]
        # self.emoticons=['emoticon-angry-outline', 'emoticon-sick-outline', 'emoticon-excited-outline', 'emoticon-neutral-outline' ,'emoticon-sad-outline','emoticon-frown-outline','emoticon-happy-outline']
        # self.emotion_icon = self.emoticons[emotion]
        # self.emotionButton=MDRoundFlatIconButton(icon=self.emotion_icon, size_hint=(0.8,width*0.8), md_bg_color=(0,0,0,0), line_color=(0,0,0,0), icon_color=(0.1,0.1,0.2,1), icon_size='240sp', z=0)
        # self.emotionOutput=MDRoundFlatIconButton(text=self.emotion_list[emotion],font_size='44sp',size_hint_x=(0.8), md_bg_color=(0,0,0,0),  line_color=(0,0,0,0), text_color=(0.1,0.1,0.2,1), halign='center', z=0)
        # self.ids.grid_emoticon.add_widget(self.emotionButton)
        # self.ids.grid_emotion.add_widget(self.emotionOutput)
        return True
    
    # def classify_upload(self):
        # self.trigger()
        # self.emotion_list=['Neutral','Happy','Sad','Anger','Fear','Disgust']
        # self.emoticons=['emoticon-happy-outline','emoticon-excited-outline','emoticon-sad-outline','emoticon-angry-outline','emoticon-frown-outline','emoticon-sick/neutral-outline']
        # self.emotion_icon = self.emoticons[self.emotion]
        # self.emotionButton=MDRoundFlatIconButton(icon=self.emotion_icon, size_hint=(0.8,width*0.8), md_bg_color=(0,0,0,0), line_color=(0,0,0,0), icon_color=(0.1,0.1,0.2,1), icon_size='240sp')
        # self.emotionOutput=MDRoundFlatIconButton(text=self.emotion_list[self.emotion],font_size='44sp',size_hint_x=(0.8), md_bg_color=(0,0,0,0),  line_color=(0,0,0,0), text_color=(0.1,0.1,0.2,1), halign='center')
        # self.ids.grid_emoticon.add_widget(self.emotionButton)
        # self.ids.grid_emotion.add_widget(self.emotionOutput)
    
    name = ''
    
    # def count(self, *varargs):
    #     self.start = datetime.now()
    #     Clock.schedule_interval(self.on_timeout, 1)

    # def on_timeout(self, *args):
    #     d = datetime.now() - self.start
    #     self.ids.top_app_bar.tile = datetime.utcfromtimestamp(d.total_seconds()).strftime("%H.%M.%S")

    def set_username(self):
        dynamodb = session.client('dynamodb')
        params = {'TableName': 'user',
                    'Key': {'userName': {'S': loginWindow.username}}}
        response = dynamodb.get_item(**params)
        item = response.get('Item', {})
        name = item.get('fullName', {}).get('S', 'unknown')
        self.ids.top_app_bar.title=name

class emotionDisplay(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(BackgroundColor())
        self.add_widget(bottomCornerBox2())
        self.add_widget(emotionCircle())
    def display_emotions(self):
        self.emotion=homeWindow.emotion
        self.emotion_list=['Neutral','Happy','Sad','Anger','Fear','Disgust']
        self.emoticons=['emoticon-happy-outline','emoticon-excited-outline','emoticon-sad-outline','emoticon-angry-outline','emoticon-frown-outline','emoticon-sick/neutral-outline']
        self.emotion_icon = self.emoticons[self.emotion]
        self.emotionButton=MDRoundFlatIconButton(icon=self.emotion_icon, size_hint=(width,width), md_bg_color=(0,0,0,0), line_color=(0,0,0,0), icon_color=(0.1,0.1,0.2,1), icon_size='320sp', halign='center')
        self.emotionOutput=MDRoundFlatIconButton(text=self.emotion_list[self.emotion],font_size='72sp',size_hint=(1,1), md_bg_color=(0,0,0,0),  line_color=(0,0,0,0), text_color=(0.1,0.1,0.2,1), halign='center')
        self.ids.grid_emoticon.add_widget(self.emotionButton)
        self.ids.grid_emotion.add_widget(self.emotionOutput)

class helpWindow(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(BackgroundColor())
        self.add_widget(bottomCornerBox())
        self.add_widget(BackgroundBox())
        

class accountWindow(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(BackgroundColor())
        self.add_widget(bottomCornerBox())
        self.add_widget(BackgroundBox())
    name,username, email, dateOfBirth = '','','',''
    def update_user_info(self):
        print(self.manager.screens)
        try:
            userN=loginWindow.username
        except:
            userN='empty'
        dynamodb = session.client('dynamodb')
        params = {'TableName': 'user',
                    'Key': {'userName': {'S': userN}}}
        response = dynamodb.get_item(**params)
        item = response.get('Item', {})
        self.name = item.get('fullName', {}).get('S', 'unknown')
        self.username = item.get('userName', {}).get('S', 'unknown')
        self.email = item.get('email', {}).get('S', 'unknown')
        self.dateOfBirth = item.get('dateOfBirth', {}).get('S', 'unknown')
        self.ids.userN.text= userN
        self.ids.Name.text = self.name
        self.ids.Email.text = self.email
        self.ids.DOB.text = self.dateOfBirth

class editProfile(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(BackgroundColor())
        self.add_widget(bottomCornerBox())
        self.add_widget(BackgroundBox())
    name,username, email, dateOfBirth = '','','',''
    def update_user_profile(self):
        try:
            userN=loginWindow.username
        except:
            userN='empty'
        dynamodb = session.client('dynamodb')
        params = {'TableName': 'user',
                    'Key': {'userName': {'S': userN}}}
        response = dynamodb.get_item(**params)
        item = response.get('Item', {})
        name = item.get('fullName', {}).get('S', 'unknown')
        self.username = item.get('userName', {}).get('S', 'unknown')
        self.email = item.get('email', {}).get('S', 'unknown')
        self.dateOfBirth = item.get('dateOfBirth', {}).get('S', 'unknown')
        self.ids.userN.text= userN
        self.ids.Name.text = name
        self.ids.Email.text = self.email
        self.ids.DOB.text = self.dateOfBirth
    def change_user_profile(self):
        pass

class CardItem(RelativeLayout):
    size_hint_y = None
    pos_hint = {'x': 0.1}
    radius = [50]
    def set_text(self, text):
        label=MDLabel(text=text, font_size='28sp')
        self.card.add_widget(label)
    def __init__(self, text='', **kwargs):
        super(CardItem, self).__init__(**kwargs)
        self.card = MDCard(md_bg_color=[0.97, 0.98, 0.99, 1],
                            radius=[30, 30, 30, 30], elevation=3)
        self.add_widget(self.card)
        self.set_text(text)

class historyWindow(Screen):
    history=DictProperty()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(BackgroundColor())
        self.add_widget(bottomCornerBox())
        self.add_widget(BackgroundBox())
        self.history={
            'r1':[3.0,'happy','18-2-2023','23:54:06'],
            'r2':[4.2,'sad','17-2-2023','22:51:36'],
            'r3':[2.1,'happy','17-2-2023','19:24:37'],
            'r4':[1.7,'angry','17-2-2023','12:31:09'],
            'r5':[1.7,'angry','17-2-2023','12:31:09']
        }
    def add_history(self): #,time,duration,emotion):
        for i in self.history.keys():
            grid = self.ids.grid
            card = CardItem()
            card.set_text(self.history[i][1])
            grid.add_widget(card)
    

class forgotPassword(Screen):
    user, newP, confirmedP = ObjectProperty(str),ObjectProperty(str),ObjectProperty(str)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(BackgroundColor())
        self.add_widget(fpWindow())

    def validateUsername(self):
        self.dynamodb = session.client('dynamodb')
        params = {'TableName': 'user',
                    'Key': {'userName': {'S': self.user.text}}}
        self.response = self.dynamodb.get_item(**params)
        try:
            self.username = self.response['Item']['userName']['S']
        except:
            self.username = 'unknown'
        if self.username == 'unknown':
            self.user.helper_text= 'Username does not exist'
            self.user.error=True
            return False
        return True
    
    def validPassword(self):
        if len(self.newP.text)>=8 and len(set(self.newP.text)&set(string.ascii_lowercase))>0 and len(set(self.newP.text)&set(string.ascii_uppercase))>0 and len(set(self.newP.text)&set(string.digits))>0:
            return True
        else: 
            self.newP.helper_text='Weak Password'
            self.newP.error=True
            return False
    def match(self):
        if self.newP.text!=self.confirmedP.text: 
            self.confirmedP.helper_text= "Passwords don't match"
            self.confirmedP.error=True
            return False
        return True
    
    def changePassword(self):
        item = self.response['Item']
        gfg = hashlib.sha3_256()
        pas = self.confirmedP.text+str(item['registrationDate&Time'])
        gfg.update(pas.encode(encoding = 'UTF-8'))
        passw = gfg.digest()
        item['password']= {'S':str(passw)}
        self.dynamodb.put_item(Item=item, TableName='user')

    def showP(self):
        if self.newP.password == True:
            self.newP.password = False

    def hideP(self):
        if self.newP.password == False:
            self.newP.password = True

    def showPassword(self):
        if self.confirmedP.password == True:
            self.confirmedP.password = False

    def hidePassword(self):
        if self.confirmedP.password == False:
            self.confirmedP.password = True

#represents the transitions between windows
class WManager(ScreenManager):
    pass


class screenApp(MDApp):
    def build(self):
        Builder.load_file('screen.kv')
        Window.size = (width, height)
        # sm= ScreenManager()
        # sm.add_widget(IntroWindow(name ="intro"))
        # sm.add_widget(createWindow(name ="signup"))
        # sm.add_widget(loginWindow(name ="login"))
        # sm.add_widget(homeWindow(name ="home"))
        # sm.add_widget(helpWindow(name ="help"))
        # sm.add_widget(accountWindow(name ="account"))
        # sm.add_widget(historyWindow(name ="history"))
        # sm.add_widget(forgotPassword(name ="forgotPassword"))
        # sm.add_widget(editProfile(name ="edit"))
        # Window.softinput_mode = 'below_target'
        # sm=WManager()
        # screens=['account','help','forgotPassword','signup','login','account','edit','intro','home','history','help']
        # for i in range(len(screens)):
        #     screen = Screen(name=screens[i])
        #     sm.add_widget(screen)
        # sm.add_widget(loginWindow(name='login'))
        # print(sm.screens)
        return 

if __name__ == '__main__':
    Config.set('kivy','keyboard_mode','systemanddock')
    screenApp().run()
