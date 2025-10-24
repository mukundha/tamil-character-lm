## Character based Tamil language model

##### Purpose
Educational

##### Goals

- Build a single layer transformer from scratch
- Train it using Thirukural dataset

#### Stats
|  |  | 
| -- | -- |
| Transformer Layers | 1 |
| Model Params |  66,432 | 
| Vocabulary Size |  64 | 
| Tokens | 640294 | 


### Prereq
```
uv pip install torch tqdm matplotlib
```

### Training 
```
uv run train.py
```

### Inference
```
uv run infer.py
```

### Model Summary
```
uv run summary.py
```


 ##### CITATION
 - Thirukural dataset downloaded from https://github.com/tk120404/thirukkural
 - Attention is all you need - https://arxiv.org/abs/1706.03762
 - ChatGPT :) 

 #### [License](./LICENSE)