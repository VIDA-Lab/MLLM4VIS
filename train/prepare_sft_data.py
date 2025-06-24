import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the JSONL file
df = pd.read_json('/path/to/your/data.jsonl', lines=True)

# Prepare the data for SFT
res = []
for index, row in df.iterrows():
    question = row.get("question")
    answer = row.get("answer")
    thought = row.get("thought")

    id = row.get("filename").split('.')[0]
    image_path = '/path/to/your/charts/old_jpg/' + row.get("filename")

    if type(thought) == str:
        answer = thought + 'ANSWER: ' + answer
        
    file_data = {
        "id": id,
        "image": image_path,
        "conversations": [
            {
                'role': 'user',
                'content': f'<image>\n{question}'
            },
            {
                'role': 'assistant',
                'content': answer
            }
        ]
    }
    res.append(file_data)
    
final_data = pd.DataFrame(res)

# Split training and evaluation data
df_train, df_eval = train_test_split(final_data, test_size=0.2, random_state=42)


df_train.to_json('/path/to/your/train.json', orient='records', indent=4, force_ascii=False)
df_eval.to_json('/path/to/your/eval.json', orient='records', indent=4, force_ascii=False)