from code import *

bert = CustomDistilBert()


def test_distilbert_optimizer():
    bert.assign_optimizer()


def test_distilbert_slice_cls_hidden_state():
    tokens = bert.tokenizer("Hi my dog is cute", return_tensors="pt")
    outputs = bert.distilbert(**tokens)
    bert.slice_cls_hidden_state(outputs)


def test_distilbert_tokenize():
    premises = ["I am wearing a hat", "Triangles are a shape"]
    hypotheses = ["There is something on my head", "Triangles are not a shape"]
    bert.tokenize(premises, hypotheses)


def test_distilbert_forward():
    premises = ["I am wearing a hat", "Triangles are a shape"]
    hypotheses = ["There is something on my head", "Triangles are not a shape"]
    encoding = bert.tokenize(premises, hypotheses)
    probs = bert.forward(encoding)
