import json
import boto3
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from io import BytesIO
import tempfile
from tensorflow.keras.models import load_model
import base64


def lambda_handler(event, context):
      ret = {}
      try: 
        s3_client = boto3.client('s3')
        m_Bucket = 'elai-processed-recordings'
        m_Name = 'model-2DCNN.h5'
        m = s3_client.get_object(Bucket=m_Bucket, Key=m_Name)['Body'].read()
        buffer = BytesIO(m)
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            temp_file.write(buffer.getvalue())
            temp_file_path = temp_file.name
        model = load_model(temp_file_path, compile=False)
        temp_file.close()
        ret.update({"model":"model loading works"})
      except: 
        ret.update({"model":"model loading didnt work"})
      try:
        dic = event["data"]
        decoded_data = base64.b64decode(dic)
        decoded_str = decoded_data.decode('utf-8')
        decoded_dict = json.loads(decoded_str)
        ret.update({"decode":"decoding works"})
      except:
        ret.update({"decode":"decoding didnt work"})
      try:
        data = np.array(decoded_dict).reshape([256,256,3])
        ret.update({"numpy":"numpy works"})
      except:
        ret.update({"numpy":"numpy didnt work"})
      try:
        prediction = model.predict(np.expand_dims(data, axis=0))
        predicted_class = np.argmax(prediction)
        ret.update({"prediction":"prediction works"})
      except:
        ret.update({"prediction":"prediction didnt work"})
      try:
        ret.update({"prediction class":int(predicted_class)})
      except:
        ret.update({"prediction class":"prediction class didnt work"})
      return {
         "body": json.dumps(ret)
      }
