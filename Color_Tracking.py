# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:54:09 2018

@author: user
"""
  

import cv2   
import numpy as np
import imutils
import time
import paho.mqtt.client as paho

from multiprocessing import Queue    #使用多核心的模組 Queue
from collections import deque
import threading, time
import os
import requests
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image, ImageFile

from scipy import ndimage

#video Streaming module
from flask import Flask, render_template, Response
#from camera import VideoCamera

def thread_video(q,client):
    print('T1 start\n')
    

    #capturing video through webcam
    cap=cv2.VideoCapture(0)
    
    while(1):
        
        success, img = cap.read()
        

    	#converting frame(img i.e BGR) to HSV (hue-saturation-value)
        if success == True: 
            #img = imutils.resize(img, width=1000)            #定義視窗長寬
            hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        else:
            continue
        
         
    	#definig the range of red color
        red_lower=np.array([0,80,100],np.uint8)
        red_upper=np.array([15,180,255],np.uint8)
        
      #definig the range of red color
        red_lower_light=np.array([0,140,70],np.uint8)
        red_upper_light=np.array([15,255,255],np.uint8)
        
        
        green_lower_light=np.array([60,190,150],np.uint8)
        green_upper_light=np.array([90,255,255],np.uint8)
        
        yellow_lower_light=np.array([15,200,30],np.uint8)
        yellow_upper_light=np.array([30,255,255],np.uint8)
        
        '''
    	#definig the range of red color
        red_lower=np.array([136,87,111],np.uint8)
        red_upper=np.array([180,255,255],np.uint8)
        '''
    	#defining the Range of Blue color
        blue_lower=np.array([99,115,150],np.uint8)
        blue_upper=np.array([110,255,255],np.uint8)
    	
    	#defining the Range of yellow color
        yellow_lower=np.array([22,60,200],np.uint8)
        yellow_upper=np.array([60,255,255],np.uint8)
    
    	#finding the range of red,blue and yellow color in the image
        red=cv2.inRange(hsv, red_lower, red_upper)
        blue=cv2.inRange(hsv,blue_lower,blue_upper)
        yellow=cv2.inRange(hsv,yellow_lower,yellow_upper)

        red_light=cv2.inRange(hsv, red_lower_light, red_upper_light)
        green_light=cv2.inRange(hsv,green_lower_light,green_upper_light)
        yellow_light=cv2.inRange(hsv,yellow_lower_light,yellow_upper_light)    	
        #Morphological transformation, Dilation  	
        kernal = np.ones((5 ,5), "uint8")
       
        red=cv2.dilate(red, kernal)
        res=cv2.bitwise_and(img, img, mask = red)
        '''    
        blue=cv2.dilate(blue,kernal)
        res1=cv2.bitwise_and(img, img, mask = blue)
    
        yellow=cv2.dilate(yellow,kernal)
        res2=cv2.bitwise_and(img, img, mask = yellow)    
        '''    
        red_light=cv2.dilate(red_light, kernal)
        res_red=cv2.bitwise_and(img, img, mask = red_light)
        
        green_light=cv2.dilate(green_light, kernal)
        res_green=cv2.bitwise_and(img, img, mask = green_light)
        
        yellow_light=cv2.dilate(yellow_light, kernal)
        res_yellow=cv2.bitwise_and(img, img, mask = yellow_light)
        
    	#Tracking the Red Color
        (_,contours,hierarchy)=cv2.findContours(red,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    	
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)          
            #print(area)
            if(area>4500):
                color_range=area
                ((x, y), radius) = cv2.minEnclosingCircle(contour)	
                cv2.circle(img, (int(x), int(y)), int(radius),(0,0,255),2)
                #cv2.putText(img,"RED color",(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))
                crop_img = img[int(y)-int(radius):int(y)+int(radius),int(x)-int(radius):int(x)+int(radius)]
                if(int(x)-int(radius)>0 and int(x)+int(radius)<500 and int(y)-int(radius)>0 and int(y)+int(radius)<500):
                    #cv2.imshow("cropped", crop_img)
                    #print(len(crop_img))
                    while (len(crop_img)>50):
                        cv2.imwrite("D:\Desktop\machine_data\\output..png",crop_img)
                        t2_start()
                        break
                    
            
        '''
        #Tracking the Blue Color
        (_,contours,hierarchy)=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area>300):
                x,y,w,h = cv2.boundingRect(contour)	
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(img,"Blue color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0))
        
    	#Tracking the yellow Color
        (_,contours,hierarchy)=cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
        	area = cv2.contourArea(contour)
        	if(area>300):
        		x,y,w,h = cv2.boundingRect(contour)	
        		img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        		cv2.putText(img,"yellow  color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))  
        '''
        #print(str(res_new)+","+str(pre_enable)+","+str(color_num))
        if(res_new==3 and pre_enable==1):
            
            (_,contours,hierarchy)=cv2.findContours(red_light,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>1500 and area<6000):
                    #print("red"+str(area))
                    color_range=area
                    ((x, y), radius) = cv2.minEnclosingCircle(contour)	
                    cv2.circle(img, (int(x), int(y)), int(radius),(0,0,255),2)
                    color_num=0
                    #cv2.putText(img,"red color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0))
                
            (_,contours,hierarchy)=cv2.findContours(green_light,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>1500 and area<6000):
                    #print("green"+str(area))
                    color_range=area
                    ((x, y), radius) = cv2.minEnclosingCircle(contour)	
                    cv2.circle(img, (int(x), int(y)), int(radius),(0,255,0),2)
                    color_num=2
                                        
            (_,contours,hierarchy)=cv2.findContours(yellow_light,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area>1500 and area<6000):
                    ((x, y), radius) = cv2.minEnclosingCircle(contour)	
                    cv2.circle(img, (int(x), int(y)), int(radius),(255,0,0),2)
                    color_num=1
            #pre_enable=0
        
        
        
        global res_before,res_new,text_a,color_range,color_num,pre_enable,output_frame

        
        label_dict={0:"20 limit sign",1:"right sign",2:"stop sign",3:"traffic light",4:"nothing"}
        
        while(q.empty()==False):
            
            res_before=q.get()
            #cv2.putText(img, "result = "+str(label_dict[q.get()]), (0,25), 0, 1, (0,255,0),2)
            
            break

        
        if(res_new!=res_before):
            res_new=res_before
            t3_start(client)
        else:
            cv2.putText(img, "result = "+str(label_dict[res_new]), (0,25), 0, 1, (0,255,0),2)
           
        	#cv2.imshow("Redcolour",red)
            
        ret, jpeg = cv2.imencode('.jpg', img)
        output_frame = jpeg.tobytes()
       
        
        #print(output_frame)
            
        cv2.imshow("Color Tracking",img)
        

        
        	#cv2.imshow("red",res) 	
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        
    print('T1 finish')
def t2_start() :
    global file_size
    fsize = os.path.getsize("D:\Desktop\machine_data"+"\\"+"output..png")
    fsize = fsize/float(1024)
    #print(fsize)
    file_size=fsize
    while(fsize>17):
           thread2 = threading.Thread(target=T2_job, args=(q,))
           thread2.start() 
           break
     
def T2_job(q):
    #print('T2 start')
    
    '''
    key = cv2.waitKey(1) & 0xFF
          
    while(1):    
        print(q.get())
        if key == ord("q"):
            break 
    '''
    global pre_enable
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    im = Image.open("D:\\Desktop\\machine_data"+"\\"+"output..png")
    img = im.resize((200,200))
    gray = img.convert('L')
    x = image.img_to_array(gray)


    x = np.expand_dims(x,axis=0)

    x = x.astype('float32')
    x /= 255

    Predicted_Probability=model.predict(x)
    #print(Predicted_Probability)
    #print (np.argmax(Predicted_Probability))
    label_pre=np.argmax(Predicted_Probability)
    q.put(label_pre)
    pre_enable=1
    
    #print('T2 finish')
def t3_start(client) :
    thread3 = threading.Thread(target=T3_job, args=(q,client))
    thread3.start() 

    

def T3_job(q,client):
    #print('T3 start') 
    text_b=q.get()
    global text_a
    #print(text_a)
    #print(text_b)
    if (text_a!=text_b):
        text_a=text_b
        #my_data = {'key': text_a}
        #r = requests.post('https://f278f6e7.ngrok.io/python1', data = my_data)
        #res=str(text_a)
        #client.publish("res",res,2)
        #print("a")    
    
    #print(r.status_code)
   
    #print('T3 finish')
    
def t4_start(client) :
    thread4 = threading.Thread(target=T4_job, args=(q,client))
    thread4.start() 
    
def T4_job(q,client):
    #print('T4 start') 

    global text_a,color_num,pre_enable,file_size
    
    all_res=car_res+","+speed+","+str(text_a)+","+str(file_size)+","+str(color_num)
    #print(all_res)
    client.publish("all_res",all_res,2)

    if(str(color_num)!="3"):
        time.sleep(5)
        color_num=3
        pre_enable=0
    #print('T4 finish')


 

def main():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to broker")
            global Connected                #Use global variable
            Connected = True                #Signal connection 
            client.subscribe("car")
            client.subscribe("speed")
            #client.subscribe("sonar")
        else:
            print("Connection failed") 
        
    def on_message(client, userdata, msg):
        #print("message topic " ,str(msg.topic))
        #print("message received " ,str(msg.payload.decode("utf-8")))
        if (str(msg.topic)=="car"):
            #print("message received = "+str(msg.payload.decode("utf-8")))
            global car_res                #Use global variable
            car_res=str(msg.payload.decode("utf-8"))
            t4_start(client)
            
        if (str(msg.topic)=="speed"):
            #print("message received = "+str(msg.payload.decode("utf-8")))
            global speed                #Use global variable
            speed=str(msg.payload.decode("utf-8"))
            t4_start(client)
            #print(speed)

        
    broker="broker.mqttdashboard.com"
    client = paho.Client()
    client.on_connect = on_connect        
    client.on_message = on_message
    client.connect(broker, 1883, 60)
    #client.loop_forever()

    client.loop_start()
    while Connected != True:    #Wait for connection
        time.sleep(1)
        print(Connected)
    
    video_job(app)
    
    thread1 = threading.Thread(target=thread_video, args=(q,client))
    thread1.start()
    
    print('all done')



#video main
app = Flask(__name__)
 
def video_job(app):
    #print('T4 start')

    @app.route('/')
    def index():
        return render_template('index.html')
        
    def gen():
        while True:
            frame = output_frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
    @app.route('/video_feed')
    def video_feed():
        return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')
    



    #print('T4 finish')


    
if __name__=='__main__':
    model = load_model('D:\Desktop\machine_data\savemodel\model1.h5')
    print('test model...')
    print(model.predict(np.zeros((1, 200,200,1))))
    print('test done.')
    q = Queue() # 開一個 Queue 物件
    #global init set 
    car_res=""
    speed=0
    text_a=3
    res_before=4
    res_new=4
    color_range=3
    color_num=4
    Connected=False
    pre_enable=0
    file_size=0
    #vidoe ip
    app.run(host='192.168.43.196', debug=True)    

    main()

   

