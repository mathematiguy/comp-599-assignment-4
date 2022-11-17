from code import *

def test_freeze_params():

    model = CustomDistilBert()
    model = freeze_params(
        model.get_distilbert()
    )

    assert not any(p.requires_grad for p in model.parameters())

def test_pad_attention_mask():
    model = CustomDistilBert()
    batch = model.tokenize(
        ["This is a premise.", "This is another premise."],
        ["This is a hypothesis.", "This is another hypothesis."],
    )
    sp = SoftPrompting(
        p=5, e=model.get_distilbert().embeddings.word_embeddings.embedding_dim
    )
    batch.input_embedded = sp(model.get_distilbert().embeddings(batch.input_ids))
    batch.attention_mask = pad_attention_mask(batch.attention_mask, 5)
