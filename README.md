# Train VisNLModel

## Environment
```bash
git clone https://github.com/OpenBMB/MiniCPM-V.git

cd MiniCPM-V 
pip install -r requirements.txt
```

## Prepare SFT Training Data
After downloading the VisNLModel dataset, you can use the following script to prepare the SFT training data.
```bash
python prepare_sft_data.py
```
Because the training data for `DataFeat Tasks` and `SimVec Tasks` are different, we only prvide one example of the `prepare_sft_data.py` script for `DataFeat Tasks`, which has the **CoT** intermediate process. Here are some examples for different SFT data formats.
```json
// DataFeat Tasks
    {
        "id":"bar-stacked_2288_4_5_multi_vertical_aspect_19_pastel2_53",
        "image":"\/data\/clda\/MLLM\/VisLM\/fine_tuning\/dataset_20241028_1742_4000\/old_jpg\/bar-stacked_2288_4_5_multi_vertical_aspect_19_pastel2_53.jpg",
        "conversations":[
            {
                "role":"user",
                "content":"<image>\nWhat is the max value of CO2 emissions for Waste Management?"
            },
            {
                "role":"assistant",
                "content":"The mapping of CO2 emissions is value 0 is at pixel 204 and value 16000 is at pixel 21, and the size of Waste Management 2010-04 is pixel 16.  ANSWER: [[16 \/ (204 - 21) * (16000 - 0)]] in 2010-04."
            }
        ]
    },
    {
        "id":"bar_89_1_5_multi_vertical_aspect_11_paired_72",
        "image":"\/data\/clda\/MLLM\/VisLM\/fine_tuning\/dataset_20241028_1742_4000\/old_jpg\/bar_89_1_5_multi_vertical_aspect_11_paired_72.jpg",
        "conversations":[
            {
                "role":"user",
                "content":"<image>\nIdentify the Gender Equality Index of Canada."
            },
            {
                "role":"assistant",
                "content":"The mapping of Gender Equality Index is value 0 is at pixel 342 and value 100 is at pixel 25, and the size of Canada is pixel 266.  ANSWER: Gender Equality Index of Canada is [[266 \/ (342 - 25) * (100 - 0)]]."
            }
        ]
    },
// SimVec Tasks
    {
        "id":"area_2614_2_8_multi_vertical_aspect_17_set3_71",
        "image":"\/data\/clda\/MLLM\/VisLM\/fine_tuning\/dataset_20241028_1742_4000\/old_jpg\/area_2614_2_8_multi_vertical_aspect_17_set3_71.jpg",
        "conversations":[
            {
                "role":"user",
                "content":"<image>\nDescribe the encoding of the chart."
            },
            {
                "role":"assistant",
                "content":"This is a area chart. The X-axis is Time (time, temporal): value 2012 at 101 and value 2018 at 415.  The Y-axis is Military Expenditure (value, quantitative): value 0 at 224 and value 600 at 30. "
            }
        ]
    },
    {
        "id":"area_1316_4_10_multi_vertical_aspect_11_pastel2_44",
        "image":"\/data\/clda\/MLLM\/VisLM\/fine_tuning\/dataset_20241028_1742_4000\/old_jpg\/area_1316_4_10_multi_vertical_aspect_11_pastel2_44.jpg",
        "conversations":[
            {
                "role":"user",
                "content":"<image>\nwhat is the simple vector format of the chart?"
            },
            {
                "role":"assistant",
                "content":"rect (0,0,20) [0,0,500,345]|text June [62,317,18,12]|text August [123,317,27,12]|text October [195,317,32,12]|text December [265,317,40,12]|text February [344,317,35,12]|text Time [218,329,22,12]|text 0 [51,307,5,12]|text 200000 [23,274,33,12]|text 400000 [23,242,33,12]|text 600000 [23,210,33,12]|text 800000 [23,178,33,12]|text 1000000 [17,146,39,12]|text 1200000 [17,113,39,12]|text 1400000 [17,81,39,12]|text 1600000 [17,49,39,12]|text 1800000 [17,17,39,12]|text Casualties [3,145,12,47]|text Civil_War [424,35,35,12]|text Cold_War [424,47,37,12]|text Revolutionary_War [424,58,73,12]|text World_War_I [424,69,47,12]|text War [410,21,18,12]|text The_Trend_of_Casualties_in_War [145,0,167,16]|line (0,0,17) 63,24;396,24;396,314;63,314;63,24|area (9,9,16) 62,134;99,124;137,128;175,125;211,116;249,115;286,100;323,74;361,86;395,38;395,50;361,86;323,85;286,100;249,116;211,116;175,125;137,136;99,145;62,149|area (1,19,17) 62,149;99,145;137,136;175,125;211,116;249,116;286,100;323,85;361,86;395,50;395,221;361,245;323,245;286,252;249,266;211,265;175,269;137,271;99,271;62,262|area (12,8,17) 62,262;99,271;137,271;175,269;211,265;249,266;286,252;323,245;361,245;395,221;395,269;361,280;323,276;286,280;249,294;211,290;175,283;137,277;99,272;62,277|area (18,13,17) 62,277;99,272;137,277;175,283;211,290;249,294;286,280;323,276;361,280;395,269;395,313;361,313;323,313;286,313;249,313;211,313;175,313;137,313;99,313;62,313|circle (9,9,16) [411,38,8,8]|circle (1,19,17) [411,49,8,8]|circle (12,8,17) [411,60,8,8]|circle (18,13,17) [411,71,8,8]"
            }
        ]
    },
```
 
Meanwhile, we also train Qwen-VL with the same data. The training SFT data is sligtly different from the MiniCPM-V SFT data. The data format is as follows:
```json
    {
        "id":"bar_2385_1_10_single_vertical_aspect_18_None_96",
        "image":"\/data\/clda\/MLLM\/VisLM\/fine_tuning\/dataset_20241028_1742_4000\/old_jpg\/bar_2385_1_10_single_vertical_aspect_18_None_96.jpg",
        "conversations":[
            {
                "from":"user",
                "value":"Picture 1: <img>\/data\/clda\/MLLM\/VisLM\/fine_tuning\/dataset_20241028_1742_4000\/old_jpg\/bar_2385_1_10_single_vertical_aspect_18_None_96.jpg<\/img>\nIdentify the revenue of Energy."
            },
            {
                "from":"assistant",
                "value":"The mapping of revenue is value 0 is at pixel 257 and value 40000 is at pixel 50, and the size of Energy is pixel 127.  So, revenue of Energy is [[127 \/ (257 - 50) * (40000 - 0)]]."
            }
        ]
    },
    {
        "id":"line_2817_4_9_multi_vertical_aspect_10_paired_53",
        "image":"\/data\/clda\/MLLM\/VisLM\/fine_tuning\/dataset_20241028_1742_4000\/old_jpg\/line_2817_4_9_multi_vertical_aspect_10_paired_53.jpg",
        "conversations":[
            {
                "from":"user",
                "value":"Picture 1:<img>\/data\/clda\/MLLM\/VisLM\/fine_tuning\/dataset_20241028_1742_4000\/old_jpg\/line_2817_4_9_multi_vertical_aspect_10_paired_53.jpg<\/img>\nDescribe the encoding of the chart."
            },
            {
                "from":"assistant",
                "value":"This is a line chart. The X-axis is Time (time, temporal): value 2025 at 77 and value 2060 at 372.  The Y-axis is Page Count (value, quantitative): value 0 at 342 and value 1000 at 26. "
            }
        ]
    },
```
## Training
Then, you can set training parameters in `MiniCPM-V/finetune/finetune_lora.sh` according to the parameters in the `MiniCPM_lora.sh` file and run it to start training.

## Inference
After training, you can use the following script to do inference with fine_tuned MiniCPM-V.
```bash
python inference_with_MiniCPM.py
```
Also, you can use the `inference_with_Qwen` script to do inference with fine_tuned Qwen-VL.

To compare with the state-of-art Vision Language Model: `GPT4o`, we provide the inference script `inference_with_GPT4o.py` using GPT4o api.

## Post processing and evaluation

You may need a some post preprocessing and manually check the final results sometimes.
All results are combined in `results.csv`.

You need to download the VisNLModel dataset from https://huggingface.co/datasets/clda/VisNLModel.