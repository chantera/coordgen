import pytest
import transformers

from coordgen.models import BertForCoordinationGeneration


@pytest.fixture(scope="module")
def model_name():
    return "bert-base-cased"


@pytest.fixture(scope="module")
def tokenizer(model_name):
    return transformers.AutoTokenizer.from_pretrained(model_name)


def test_embed_mask(tokenizer):
    raw = "Gold will retain its gain, he said."
    s1, s2 = BertForCoordinationGeneration.embed_mask(tokenizer, raw, (10, 25), "and")
    assert s1 == "Gold will retain its gain and [MASK] [MASK] [MASK], he said."
    assert s2 == "Gold will [MASK] [MASK] [MASK] and retain its gain, he said."


def test_embed_coord():
    raw = "Gold will retain its gain, he said."
    s, coord = BertForCoordinationGeneration.embed_coord("rise further", raw, (10, 25), "and")
    assert s == "Gold will retain its gain and rise further, he said."
    assert len(coord.conjuncts) == 2
    conj1, conj2 = coord.conjuncts
    assert s[conj1[0] : conj1[1]] == "retain its gain"
    assert s[conj2[0] : conj2[1]] == "rise further"
