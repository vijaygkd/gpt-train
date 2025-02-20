# gpt-train
Train a GPT-2 124M model

Reference:
* Karpathy youtube video - [Let's reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU&t=1166s)
* Karpathy repo - https://github.com/karpathy/build-nanogpt


## 50B Training Run

* Model size: 124M
* Training Tokens: 50B
* Dataset: Fineweb-edu-10B sample
* Epochs: 5

**Results:**
* Min Train Loss: 2.6886
* Min Validation Loss: 2.9401
* Max Hellaswag eval: 0.3353

The model outperformed GPT-2 model (10B) and matched GPT-3 model (300B tokens) performance on Hellaswag eval with much less training data.
  
![image](https://github.com/user-attachments/assets/e8384066-1da7-4a68-9b12-acf97ed3c18e)
