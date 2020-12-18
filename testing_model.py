import os
import cv2
import numpy as np
from keras.models import model_from_json
import boto3
from botocore.config import Config
import base64
import json
import PIL
import PIL.Image
from io import BytesIO
from keras.preprocessing import image as keras_image
import json

aws_access_key_id = ""
aws_secret_access_key = ""

with open('config/aws_credentials.json') as f:
	data = json.load(f)
	aws_access_key_id = data["aws_access_key_id"]
	aws_secret_access_key = data["aws_secret_access_key"]

model = model_from_json(open("data/model.json", "r").read())
model.load_weights('data/weight.h5')

my_config = Config(region_name = 'us-east-2',)
client = boto3.client('sagemaker-runtime', config=my_config, 
						aws_access_key_id=aws_access_key_id,
						aws_secret_access_key=aws_secret_access_key)

bucket_name="projectmtestset"
s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)

emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
image_size = 256
face_haar_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

for i in range(len(emotions)):
	for obj in bucket.objects.filter(Prefix=("Testing_Emotion/"+emotions[i]+"/")):
		if obj.size:
			body = obj.get()['Body'].read()
			image = PIL.Image.open(BytesIO(body)).convert('RGBA')
			image_array = np.array(image)
			gray_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
			faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

			for (x,y,w,h) in faces_detected:
				cv2.rectangle(image_array,(x,y),(x+w,y+h),(255,0,0),2)
				roi_gray=gray_img[y:y+w,x:x+h]
				roi_gray=cv2.resize(roi_gray,(image_size,image_size))
				img_pixels = keras_image.img_to_array(roi_gray)
				img_pixels = np.expand_dims(img_pixels, axis = 0)
				img_pixels /= 255
				
				predictions = model.predict(img_pixels)
				max_index = np.argmax(predictions[0])
				predicted_emotion = emotions[max_index]
				print(obj.key + ": " + predicted_emotion)
