## Character based Tamil language model

### Purpose
Educational

### Goals

- Build a single layer transformer
- Train it using Thirukural dataset
- Run it on a laptop in few minutes

#### Stats
| Key | Value | 
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

#### Expected Results

- After 10 epochs (loss 1.6673)
```
திரும் தெறிக்கும் நீரு என்போது அறிவும் சொல்படங்கிய தாய்ந்குத் தென்னதெனத்து ஒருவன்.
கருணாநிதி: காள் குலகும் வலியாசத்து நற்கி வோது வர்.
வருக்குறள்: பல்லவற்று இன்றையை அனியுமை ஒல்லாத அச்சிய பிடப்போல், இவ்வதற்
```

- After 50 epochs (loss 1.3316)
```
திருந்தாமல் வேண்டார்களுடியின் வாழ்வாறு யாத்துச் சிறந்ததில் பேசாவினால் வுளையே மட்டும்.
திருக்குறள்: ஒருவன் வேண்டு மேனக் கோலைப் ப் பின்றுச்சிகிறப்பும் ஒன்றராப் காகை. தவரேன் இக்கத்தலை.
வரதராய்தரிருகாலேம் தோ,
```

 ##### CITATION
 - Thirukural dataset downloaded from https://github.com/tk120404/thirukkural
 - Attention is all you need - https://arxiv.org/abs/1706.03762
 - ChatGPT :) 

 #### [License](./LICENSE)