import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
from transformers.generation import GenerationConfig
from tqdm import tqdm



# Load the lora model and tokenizer
path_to_adapter="/path/to/your/output_dir/checkpoint-***" 

model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    cache_dir=path_to_adapter,
    trust_remote_code=True
).eval()
model.generation_config = GenerationConfig.from_pretrained("/data1/model_checkpoint/Qwen-VL-Chat", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(path_to_adapter, trust_remote_code=True)



# Load your testset and run inference
df = pd.read_json('/path/to/your/test.json')


for index, row in tqdm(df.iterrows(), total=len(df)):
    image_path = '/path/to/your/charts/old_jpg/' + row.get("id") + '.jpg'
    origin_question = row.get("question")
    
    query = tokenizer.from_list_format([
        {'image': image_path},
        {'text': origin_question},
    ])
    response, _ = model.chat(tokenizer, query=query, history=None, do_sample=True, temperature=0.03)
    
    df.at[index, 'prediction'] = response     
    
df.to_csv('/path/to/your/prediction.csv', index=False)
    
