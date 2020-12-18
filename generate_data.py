import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image as keras_image
import PIL
import PIL.Image
from io import BytesIO
import sys
import numpy as np
import boto3
import cv2
from botocore.config import Config
import json

aws_access_key_id = ""
aws_secret_access_key = ""

with open('config/aws_credentials.json') as f:
	data = json.load(f)
	aws_access_key_id = data["aws_access_key_id"]
	aws_secret_access_key = data["aws_secret_access_key"]

my_config = Config(region_name = 'us-east-2')

client = boto3.client('sagemaker-runtime', 
						config=my_config, 
						aws_access_key_id=aws_access_key_id,
						aws_secret_access_key=aws_secret_access_key)

bucket_name= 'emotion-testing-data'
s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)

emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
images_array = [[],[],[],[],[],[],[],[]]
images_size = [0,0,0,0,0,0,0,0]

total_images = 0

face_haar_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

for i in range(len(emotions)):
	for obj in bucket.objects.filter(Prefix=("Emotion/"+emotions[i]+"/")):
		if obj.size:
			body = obj.get()['Body'].read()
			image = PIL.Image.open(BytesIO(body)).convert('RGBA')
			image_array = np.array(image)
			gray_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
			faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

			for (x,y,w,h) in faces_detected:
				cv2.rectangle(image_array,(x,y),(x+w,y+h),(255,0,0),2)
				roi_gray=gray_img[y:y+w,x:x+h]
				roi_gray=cv2.resize(roi_gray,(256,256))
				img_pixels = keras_image.img_to_array(roi_gray)
				img_pixels = np.expand_dims(img_pixels, axis = 0)

				flat_arr = img_pixels.ravel()
				
				images_array[i].append(str(i) + ',' + ' '.join(str(int(num)) for num in flat_arr))
				images_size[i] += 1
				total_images += 1

print("Total Processed Images: {}".format(total_images))

TRAINING_RATIO = 0.95

with open('data/dataset.csv', 'w') as f:
	print('emotion,pixels,usage', file=f)

	for i in range(len(images_array)):
		num_trains = int(images_size[i] * TRAINING_RATIO)
		for j in range(len(images_array[i])):
			if j < num_trains:
				print(images_array[i][j] + ',' + 'Training', file = f)
			else:
				print(images_array[i][j] + ',' + 'Testing', file = f)
