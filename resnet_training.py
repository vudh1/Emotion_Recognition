import tflearn
import numpy as np
from tflearn.data_preprocessing import ImagePreprocessing
import pandas as pd

image_size = 256
num_labels = 8

df = pd.read_csv('data/dataset.csv')

X_train,Y_train,X_test,Y_test=[],[],[],[]

for index, row in df.iterrows():
		
		val=row['pixels'].split(" ")
		
		try:
				if 'Training' in row['usage']:
					 X_train.append(np.array(val,'float32'))
					 Y_train.append(row['emotion'])
				elif 'Testing' in row['usage']:
					 X_test.append(np.array(val,'float32'))
					 Y_test.append(row['emotion'])
		except:
				print('error occured at index {}'.format(index))


X_train = np.array(X_train,'float32')
Y_train = np.array(Y_train,'float32')
X_test = np.array(X_test,'float32')
Y_test = np.array(Y_test,'float32')