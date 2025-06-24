import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm



# Load the lora model and tokenizer
path_to_adapter="/path/to/your/output_dir/checkpoint-***" 

peft_model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
).eval()

peft_tokenizer = AutoTokenizer.from_pretrained(path_to_adapter, trust_remote_code=True)
vpm_resampler_embedtokens_weight = torch.load(f"{path_to_adapter}/vpm_resampler_embedtokens.pt")
msg = peft_model.load_state_dict(vpm_resampler_embedtokens_weight, strict=False)


# Load your testset and run inference
df = pd.read_json('/path/to/your/test.json')

for index, row in tqdm(df.iterrows(), total=len(df)):
    image_path = '/path/to/your/charts/old_jpg/' + row.get("id") + '.jpg'
    image = Image.open(image_path).convert('RGB')
    conversation = row['conversations']
    origin_question = conversation[0].get('content').replace('<image>\n', '')
    
    msgs = [{'role': 'user', 'content': origin_question}]
    model_res = peft_model.chat(
        image=image,
        msgs=msgs,
        tokenizer=peft_tokenizer,
        sampling=True, # if sampling=False, beam_search will be used by default
        temperature=0.03,
        # system_prompt='' # pass system_prompt if needed
    )
    df.at[index, 'prediction'] = model_res     
    
df.to_csv('/path/to/your/prediction.csv', index=False)
    
