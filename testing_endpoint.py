import boto3
from botocore.config import Config
import base64
import json

aws_access_key_id = ""
aws_secret_access_key = ""

with open('config/aws_credentials.json') as f:
	data = json.load(f)
	aws_access_key_id = data["aws_access_key_id"]
	aws_secret_access_key = data["aws_secret_access_key"]

endpoint_name='IC-Emotion-1606685367'
my_config = Config(
	region_name = 'us-east-2',
)
client = boto3.client('sagemaker-runtime',
						config=my_config, 
						aws_access_key_id=aws_access_key_id,
						aws_secret_access_key=aws_secret_access_key)

bucket_name="projectmtestset"
s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_name)

emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


for obj in bucket.objects.filter(Prefix="Testing_Emotion/"):
	if obj.size:
		body = obj.get()['Body'].read()
		data = {}
		data['data'] = base64.encodebytes(body).decode('utf-8')
		json_image = json.dumps(data)
		byte_image = bytearray(body)    
		response = client.invoke_endpoint(EndpointName = endpoint_name, ContentType = 'image/png', Body = byte_image)


		if response != None:
			result = response['Body'].read().decode('ascii')
			scores  = json.loads(result)
			
			max_emotion = ''
			max_score = 0.0
			
			for i in range(0,8):
				if max_score < float(scores[i]):
					max_score = float(scores[i])
					max_emotion = emotions[i]  
			
			print("{}: {}".format(obj.key, max_emotion, max_score))
