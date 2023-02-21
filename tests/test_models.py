import coordgen.models._utils as modeling_utils


def test_embed_mask():
    raw = "Gold will retain its gain, he said."
    s1, s2 = modeling_utils.embed_mask("<mask>", raw, (10, 25), "and")
    assert s1 == "Gold will retain its gain and <mask>, he said."
    assert s2 == "Gold will <mask> and retain its gain, he said."


def test_embed_coord():
    raw = "Gold will retain its gain, he said."
    s, coord = modeling_utils.embed_coord("rise further", raw, (10, 25), "and")
    assert s == "Gold will retain its gain and rise further, he said."
    assert len(coord.conjuncts) == 2
    conj1, conj2 = coord.conjuncts
    assert s[conj1[0] : conj1[1]] == "retain its gain"
    assert s[conj2[0] : conj2[1]] == "rise further"
