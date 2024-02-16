# DL11_Text_Classification
The 11th assignement of DeepLearning course- Text classification with emoji

"❤️", "⚾️", "😄", "😔", "🍴"

## How to install

```
pip install -r requirements.txt
```

##  How to run

```
python emoji_text_classification.py
```

## Results:

### Results before adding dropout layer:

| Feature vector dimension  | Train Loss | Train Accuracy  | Test Loss | Test Accuracy  | Inference Time (s) |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 50d  | 0.31  | 0.94  | 0.39 | 0.87  | 0.077  |
| 100d  | 0.16  | 0.97  | 0.42  | 0.78  | 0.059  |
| 200d  | 0.12  | 0.99  | 0.43  | 0.80  | 0.060  |
| 300d  | 0.12  | 1.00  | 0.43  | 0.85  | 0.067  |


### Results after adding dropout layer with a dropout rate 0.2:

| Feature vector dimension  | Train Loss | Train Accuracy  | Test Loss | Test Accuracy  | Inference Time (s) |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 50d  | 0.50  | 0.84  | 0.46 | 0.85  | 0.074  |
| 100d  | 0.40  | 0.88  | 0.48  | 0.82  | 0.072  |
| 200d  | 0.19  | 0.95  | 0.42  | 0.80  | 0.070  |
| 300d  | 0.10  | 1.00  | 0.39  | 0.82  | 0.101  |