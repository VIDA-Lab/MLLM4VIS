import pandas as pd
import requests
import json
import base64
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


df = pd.read_json('/path/to/your/test.json')

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def extract(response):
    match = re.search(r'"content":\s*"(.*?)"', response.text)

    if match:
        content = match.group(1)
        return content
    else:
        return "No match" + response.text

def process(question, base64_image, url):
   payload = json.dumps({
      "model": "gpt-4o",
      "messages": [
         {
            "role": "user",
            "content": [
               {
                  "type": "text",
                  "text": f"{question}",
               },
               {
                  "type": "image_url",
                  "image_url": {
                     "url": f"data:image/jpeg;base64,{base64_image}"
                  }
               }
            ]
         }
      ],
      "max_completion_tokens": 2048,
      "stream": False
   })
   headers = {
      'Authorization': 'Bearer YOUR_API_KEY',
      'Content-Type': 'application/json'
   }

   response = requests.request("POST", url, headers=headers, data=payload)

   return extract(response)

url = 'https://api.openai.com/v1/chat/completions'

for index, row in tqdm(df.iterrows(), total=len(df)):
    image_path = row.get('image')
    base64_image = encode_image(image_path)  
    question = row.get('question')
    
    model_res = process(question, base64_image, url)
    df.at[index, 'gpt4o'] = model_res   
    
df.to_csv('/path/to/your/prediction.csv', index=False)
    
    