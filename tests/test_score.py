from diseval.score import evaluate, match_sentence, normalise, prf


def test_normalise_ignores_case_and_punctuation():
    assert normalise("BRCA1 - related  Cancer") == normalise("brca1-related cancer")


def test_exact_match():
    m = match_sentence(["breast cancer"], ["Breast Cancer"])
    assert (m.tp, m.fp, m.fn) == (1, [], [])


def test_strict_rejects_boundary_difference_but_lenient_accepts():
    gold, pred = ["adenomatous polyposis coli tumour"], ["adenomatous polyposis coli"]
    strict = match_sentence(gold, pred, lenient=False)
    lenient = match_sentence(gold, pred, lenient=True)
    assert strict.tp == 0 and strict.fp == pred and strict.fn == gold
    assert lenient.tp == 1


def test_lenient_does_not_match_unrelated_words():
    assert match_sentence(["colon cancer"], ["cancer of the colon"], lenient=True).tp == 0


def test_one_to_one_matching():
    m = match_sentence(["cancer"], ["cancer", "cancer"])
    assert m.tp == 1 and m.fp == ["cancer"]


def test_exact_matches_are_made_before_lenient_ones():
    m = match_sentence(["breast cancer", "cancer"], ["cancer"], lenient=True)
    assert m.tp == 1 and m.fn == ["breast cancer"]


def test_prf_values_and_zero_division():
    s = prf(tp=8, fp=2, fn=8)
    assert round(s["precision"], 2) == 0.8 and round(s["recall"], 2) == 0.5
    assert prf(0, 0, 0)["f1"] == 0.0


def test_evaluate_counts_sentences_without_disease():
    rows = [
        {"id": "0", "text": "t0", "gold": ["asthma"], "pred": ["asthma"], "error": None},
        {"id": "1", "text": "t1", "gold": [], "pred": ["fever"], "error": None},
        {"id": "2", "text": "t2", "gold": ["gout"], "pred": [], "error": "ValueError: x"},
    ]
    scores, errors = evaluate(rows)
    assert (scores["strict"]["tp"], scores["strict"]["fp"], scores["strict"]["fn"]) == (1, 1, 1)
    assert scores["failed_calls"] == 1 and scores["sentences_with_disease"] == 2
    assert {e["kind"] for e in errors} == {"false_positive", "missed"}
