from code import *

bert = CustomDistilBert()

def test_distilbert_optimizer():
    bert.assign_optimizer()

def test_distilbert_slice_cls_hidden_state():
    tokens = bert.tokenizer("Hi my dog is cute", return_tensors="pt")
    outputs = bert.distilbert(**tokens)
    bert.slice_cls_hidden_state(outputs)

def test_distilbert_tokenize():
    premise = ["I am wearing a hat"]
    hypothesis = ["There is something on my head"]
    bert.tokenize(premise, hypothesis)
