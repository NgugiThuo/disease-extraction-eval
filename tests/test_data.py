import pandas as pd

from diseval.data import detokenize, load_split, spans_from_tags

# The example row from the dataset card
TOKENS = ['Identification', 'of', 'APC2', ',', 'a', 'homologue', 'of', 'the', 'adenomatous',
          'polyposis', 'coli', 'tumour', 'suppressor', '.']
TAGS = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0]


def test_spans_from_example():
    assert spans_from_tags(TAGS) == [(8, 12)]


def test_two_adjacent_mentions():
    assert spans_from_tags([1, 2, 1, 0, 1]) == [(0, 2), (2, 3), (4, 5)]


def test_stray_inside_tag_starts_a_span():
    assert spans_from_tags([0, 2, 2, 0]) == [(1, 3)]


def test_span_running_to_the_end():
    assert spans_from_tags([0, 1, 2]) == [(1, 3)]


def test_no_mentions():
    assert spans_from_tags([0, 0, 0]) == []


def test_detokenize_fixes_punctuation_spacing():
    assert detokenize(TOKENS) == (
        "Identification of APC2, a homologue of the adenomatous polyposis coli tumour suppressor."
    )
    assert detokenize(["gene", "(", "BRCA1", ")", "."]) == "gene (BRCA1)."


def test_load_split_roundtrip(tmp_path):
    path = tmp_path / "sample.parquet"
    pd.DataFrame({"id": ["0", "1"], "tokens": [TOKENS, ["No", "disease", "here", "."]],
                  "ner_tags": [TAGS, [0, 0, 0, 0]]}).to_parquet(path)
    examples = load_split(str(path))
    assert examples[0].gold == ["adenomatous polyposis coli tumour"]
    assert examples[1].gold == []
    assert examples[1].text == "No disease here."
    assert len(load_split(str(path), limit=1)) == 1
