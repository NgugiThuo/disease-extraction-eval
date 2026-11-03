# Disease mention extraction with an LLM: a small evaluation

How well does an LLM find disease names in biomedical text, measured against human annotations?

This repo sends sentences from the NCBI Disease corpus to an LLM through the Anthropic API,
checks the structured output against a schema, and scores it against the gold annotations.
The point is the evaluation: a measured result, with the failures analysed, not just an API call.

> **Status:** complete as a small evaluation. The v1 prompt was evaluated on the validation
> split and, once, on the test split. The prompt was not changed after seeing any results.

## Data

- **NCBI Disease corpus**, read from the Hugging Face copy (`ncbi/ncbi_disease`). Each example is
  a single sentence with word tokens and BIO tags (0 = outside, 1 = start of a disease mention,
  2 = inside one).
- Many sentences contain no disease at all (435 of the 924 validation sentences), which is
  useful for measuring false positives.
- The corpus is a US Government work with no restrictions on use (see `DATA_NOTICE.md`). The
  Hugging Face metadata lists the licence as "unknown", but the notice distributed with the data
  is the public domain notice reproduced in that file.
- The data files are not committed. Download the `validation` and `test` parquet files from the
  dataset's `refs/convert/parquet` branch on Hugging Face and save them as
  `data/validation.parquet` and `data/test.parquet`.
- **Citations** (please cite the authors in any work based on this material):
  1. An improved corpus of disease mentions in PubMed citations: http://aclweb.org/anthology-new/W/W12/W12-2411.pdf
  2. NCBI Disease Corpus: A Resource for Disease Name Recognition and Normalization: http://www.ncbi.nlm.nih.gov/pubmed/24393765
  3. Disease Name Normalization with Pairwise Learning to Rank: http://www.ncbi.nlm.nih.gov/pubmed/23969135

## Method

1. `diseval/data.py` rebuilds each sentence from its tokens and turns the BIO tags into gold mention strings.
2. `diseval/extract.py` asks the model to call a `record_diseases` tool with the mentions it finds.
   The tool call is forced, and the result is validated with Pydantic. A failed call is recorded
   and counted, never silently dropped.
3. `diseval/score.py` compares predictions with gold mentions and reports micro-averaged
   precision, recall and F1 under two matching rules:
   - **strict**: the mention must equal a gold mention after lowercasing and removing punctuation.
   - **lenient**: also accepts a mention that contains, or is contained in, a gold mention
     (for example "adenomatous polyposis coli" against "adenomatous polyposis coli tumour").

**Protocol.** The test split was run once, with the prompt fixed from the start. The prompt was
never edited after seeing validation or test results. Raw model outputs are saved in `results/`
so every score can be recomputed. Model outputs can vary slightly between runs.

## Run it

```
python -m venv .venv
.venv\Scripts\activate            # on Mac or Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest
```

1. Copy `.env.example` to `.env` and add your Anthropic API key. Never commit `.env`.
2. Put `validation.parquet` and `test.parquet` in `data/`.
3. Try five sentences first, then check the cost on your API usage page:
   ```
   python -m diseval.extract --data data/validation.parquet --out results/val_raw.jsonl --limit 5
   python -m diseval.score results/val_raw.jsonl
   ```
4. Run a full split and score it:
   ```
   python -m diseval.extract --data data/test.parquet --out results/test_raw.jsonl
   python -m diseval.score results/test_raw.jsonl --errors results/test_errors.csv
   ```

`extract` skips sentence ids that are already in the output file, so a stopped run can be resumed
by running the same command again. A failed call is saved with an `error` field. Remove those
rows before resuming if you want them retried, and check that the score output reports
`Failed calls: 0` before trusting the numbers.

## Results

Model: claude-haiku-5-5. Prompt version: v1, never changed after seeing any results.
The test split was run once.

| Split      | Match   | Precision | Recall | F1    |
|------------|---------|-----------|--------|-------|
| Validation | strict  | 0.779     | 0.701  | 0.738 |
| Validation | lenient | 0.908     | 0.818  | 0.861 |
| Test       | strict  | 0.815     | 0.708  | 0.758 |
| Test       | lenient | 0.921     | 0.800  | 0.856 |

Validation: 924 sentences, 787 gold mentions. Test: 941 sentences, 960 gold mentions.
Strict requires the same text as the annotators. Lenient also accepts a prediction that
contains, or is contained in, a gold mention. In the validation run, 77 sentences were
rerun with the same prompt after a network drop caused connection errors.

The two splits agree closely. On both, recall is lower than precision: after lenient matching,
about 20% of gold mentions on the test split still have no overlapping prediction, against about
8% of predictions that match no gold mention.

## Error analysis

I read a random sample of 40 of the 392 strict errors on the validation split (24 misses,
16 false positives, seed 0) and checked gold against predictions for the sentences behind the
most frequent patterns. The sample is small, so the groups show what kinds of error occur, not
how often. The errors are spread out: the 20 most frequent misses make up 37% of all misses.

What the model misses, partly because of my prompt:
- Disease words inside gene names: VHL in "VHL mRNA" and "tumour" in "tumour suppressor gene".
  The prompt told the model not to list genes, and the annotators marked these. VHL is missed 17
  times and tumour/tumor/tumors 23 times.
- Symptoms ("short stature", "severely handicapped", "blindness").
- Repeated mentions: in the sentences I checked, a mention that occurs twice (for example DM) was
  listed once, but the scorer counts every occurrence.

Where the model itself diverges:
- Coordinated phrases are handled inconsistently. It kept "breast and ovarian cancer" whole in
  one sentence and split it in another. In "classical and Duarte galactosemia" it split the
  phrase and produced "classical galactosemia", a string that is not in the sentence, although
  the prompt asked for exact copies.
- Modifiers are dropped or kept inconsistently ("sporadic renal cell carcinoma").
- It merged a mention and its bracketed abbreviation into one string ("familial Mediterranean
  fever (FMF)") where the gold has two. Bracketed abbreviations were otherwise handled correctly.
- It listed a generic phrase ("the condition") where the annotators marked the specific symptoms.

The prompt was not changed after these results. A prompt that states the conventions above would
probably score higher; testing that means tuning on the validation split and running the test
split again, so I report this version and leave that as an extension.

## Limitations

- The corpus has been public for years, so the model may have seen it during training. High scores
  could partly reflect memorisation and are not a guarantee on new documents.
- The gold labels are a consensus of human annotators, so some disagreements are judgement calls,
  not model errors.
- Scores are at sentence level on one corpus and one model. They do not show how the approach
  works on other kinds of text.
- Mention matching is by normalised text, not by character position, so a mention repeated in a
  sentence is counted by occurrence.
- The error analysis is based on a small sample and some targeted checks, not an annotation of
  every error.

## Possible extensions

- Link each mention to a standard identifier (MeSH or OMIM), which is the corpus's second task.
- Rewrite the prompt to state the annotation conventions, tune it on the validation split, and
  run the test split again to compare versions.
- Compare models or prompts on the same split, and add retries or confidence checks.
