# Financial Sentiment Analyzer

A Python library for classifying financial text as `bullish`, `neutral`, or
`bearish`. It can analyze text supplied by your application, online articles,
local documents, and remote documents. A Streamlit dashboard is also included.

## Features

- Analyze plain financial text
- Extract and analyze online articles
- Analyze local or remote `.txt`, `.md`, and `.pdf` files
- Return a sentiment label, sentiment score, and market direction
- Generate SHAP explanations and transformer attention visualizations
- Use the library from any Python project or run the included dashboard

## Installation

The PyPI distribution name is `financial-sentiment-analyzer`, while the Python
import name is `fin_sentiment`.

After the package has been published to PyPI, install it with:

```bash
python -m pip install financial-sentiment-analyzer
```

To install the current source directly from a cloned project:

```bash
git clone https://github.com/ademicho123/financial_sentiment_analysis.git
cd financial_sentiment_analysis
python -m pip install .
```

The first analysis downloads the TinyBERT model through Transformers and stores
it in the normal Hugging Face cache. Later calls reuse the cached model.

## Quick Start

```python
from fin_sentiment import analyze_text

result = analyze_text(
    "The company exceeded its earnings forecast and increased its dividend."
)

print(result["sentiment"])
print(result["score"])
print(result["direction"])
```

## Analyze Text from Your Application

Use `analyze_text()` when your project already has the text, such as text from a
database, API response, form, or message queue.

```python
from fin_sentiment import analyze_text


def classify_announcement(announcement: str) -> str:
    result = analyze_text(announcement)
    return result["sentiment"]


sentiment = classify_announcement(
    "Revenue declined and management lowered its full-year guidance."
)
print(sentiment)
```

The input must be a non-empty string. Invalid input raises `ValueError`.

## Analyze an Article URL

Use `analyze_article()` with a public HTTP or HTTPS article URL:

```python
from fin_sentiment import analyze_article

result = analyze_article("https://example.com/markets/company-results")

print(result["sentiment"])
print(result["score"])
```

The function downloads the HTML, removes common page elements such as scripts,
navigation, headers, and footers, and analyzes the remaining readable text.

Some websites block automated requests or render their articles entirely with
JavaScript. Those pages may need to be fetched separately before their text is
passed to `analyze_text()`.

## Analyze a Local File

Use `analyze_file()` with a path relative to your project's working directory or
with an absolute path:

```python
from fin_sentiment import analyze_file

result = analyze_file("reports/annual-results.pdf")
print(result["sentiment"])
```

Using `pathlib` also works:

```python
from pathlib import Path

from fin_sentiment import analyze_file

report = Path(__file__).parent / "reports" / "annual-results.pdf"
result = analyze_file(str(report))
```

## Analyze a Remote File

Pass a direct HTTP or HTTPS file link to `analyze_file()`:

```python
from fin_sentiment import analyze_file

result = analyze_file("https://example.com/reports/company-results.pdf")
print(result["direction"])
```

Supported file types are:

- `.txt`
- `.md`
- Text-based `.pdf`

Local and remote files are limited to 20 MB. Image-only or scanned PDFs must be
processed with OCR before they can be analyzed.

## Result Format

The analysis functions return a dictionary with these values:

| Key | Description |
| --- | --- |
| `sentiment` | Model classification: `bullish`, `neutral`, or `bearish` |
| `score` | VADER compound sentiment score between `-1` and `1` |
| `direction` | Direction derived from the sentiment score |
| `preprocessed_text` | Cleaned text retained for inspection and downstream use |
| `shap_values` | SHAP explanation object |
| `attention_plot` | Matplotlib attention visualization |

For example:

```python
result = analyze_text("The business reported record quarterly profit.")

summary = {
    "sentiment": result["sentiment"],
    "score": result["score"],
    "direction": result["direction"],
}
print(summary)
```

## SHAP Plot

Use `generate_shap_plot()` with the `shap_values` returned by an analysis:

```python
from fin_sentiment import analyze_text, generate_shap_plot

result = analyze_text("Operating costs increased sharply during the quarter.")
plot = generate_shap_plot(result["shap_values"])
```

## Error Handling

Normal Python exceptions are raised when an input cannot be processed:

```python
from fin_sentiment import analyze_article, analyze_file

try:
    result = analyze_file("reports/missing.pdf")
except FileNotFoundError as error:
    print(error)

try:
    result = analyze_article("https://example.com/unavailable-article")
except (ValueError, OSError) as error:
    print(f"Could not analyze article: {error}")
```

Remote requests time out after 20 seconds. Unsupported formats, empty files,
oversized files, and PDFs without extractable text raise `ValueError`.

## Optional Dashboard

Install the project with its dashboard dependencies:

```bash
python -m pip install ".[dashboard]"
streamlit run dashboard.py
```

The dashboard allows users to paste text or upload a PDF and inspect the result
interactively.

## Development

Install an editable copy with development tools:

```bash
python -m pip install -e ".[dev]"
```

Install the optional model-training dependencies when working on training code:

```bash
python -m pip install -e ".[training,dev]"
```

Build the wheel and source distribution with:

```bash
python -m build
```

### Running the Benchmark

Install the benchmark dependencies and run a small Financial PhraseBank check:

```bash
python -m pip install -e ".[benchmark]"
python benchmarks/benchmark_fin_sentiment.py --max-samples 100
```

Remove `--max-samples 100` to evaluate the complete all-agree test split.
Results are written to
`benchmarks/results/financial_phrasebank/` as JSON, CSV, and Markdown files.
The first run can take several minutes while TinyBERT and the required NLTK data
are downloaded and cached. Later runs reuse those cached files. The benchmark
uses `explain=False`, so SHAP and attention plots do not slow down each sample.
Limited runs use a deterministic, label-stratified sample rather than the first
rows of the dataset. Omit `--max-samples` for the authoritative full test score.

#### Verified Benchmark Results

The packaged model was evaluated on the Financial PhraseBank all-agree test
split using Python 3.12.10:

| Evaluation | Samples | Accuracy | Macro F1 | Runtime |
| --- | ---: | ---: | ---: | ---: |
| Stratified quick check | 100 | 0.9700 | 0.9542 | 38 seconds |
| Full held-out test split | 453 | 0.9404 | 0.9220 | 56 seconds |

The 453-example result is the authoritative benchmark. Its weighted F1 is
0.9408. Per-class results were:

| Sentiment | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Bearish | 0.8816 | 0.9178 | 0.8993 | 73 |
| Neutral | 0.9745 | 0.9571 | 0.9658 | 280 |
| Bullish | 0.8922 | 0.9100 | 0.9010 | 100 |

The complete metrics, classification report, confusion matrix, and individual
predictions are generated under `benchmarks/results/financial_phrasebank/`.

### Publishing to PyPI

Releases are published through GitHub Actions using PyPI Trusted Publishing; no
PyPI API token is stored in the repository.

Configure the pending publisher in PyPI with:

- PyPI project name: `financial-sentiment-analyzer`
- GitHub owner: `ademicho123`
- Repository name: `financial_sentiment_analysis`
- Workflow name: `release.yml`
- Environment name: `pypi`

In the GitHub repository, create an environment named `pypi` under **Settings →
Environments**. Requiring a maintainer's approval for this environment is
recommended.

To publish a release:

1. Update `version` in `pyproject.toml` to a version that has never been uploaded.
2. Commit and push the version change.
3. Create a GitHub release for that commit with a matching tag such as `v0.1.0`.
4. Publish the GitHub release.
5. Approve the `pypi` environment deployment if approval protection is enabled.
6. Check the workflow under the repository's **Actions** tab.

Publishing the GitHub release triggers `.github/workflows/release.yml`, which
builds, validates, and uploads the wheel and source distribution to PyPI.

## Model

The project uses `huawei-noah/TinyBERT_General_4L_312D` for classification and
also provides VADER scoring, SHAP explanations, and attention visualization.
The project experiments additionally cover AdaBoost, SVM, random forest, naive
Bayes, logistic regression, and neural-network models.

The packaged TinyBERT classification head is fine-tuned on the all-agree
Financial PhraseBank training split. Reproduce the training with:

```bash
python -m pip install -e ".[training]"
python training/train_tinybert.py
```

Training saves the model, tokenizer, label mapping, and metrics under
`fin_sentiment/model/`. Inference loads this trained local checkpoint rather
than creating a random classification head from the base model.

The packaged version was trained for three epochs with seed 42. On the held-out
453-example Financial PhraseBank all-agree test split it achieved 94.04%
accuracy, 0.9220 macro F1, and 0.9408 weighted F1. Full settings and per-epoch
results are recorded in `fin_sentiment/model/training_metrics.json`.

## Contributing

Contributions are welcome from everyone. If you find a bug, have a feature idea,
or want to suggest an improvement, please raise an issue in the GitHub
repository. Pull requests for fixes, documentation, tests, and new features are
also welcome.

To contribute code:

1. Fork the repository and create a branch for your change.
2. Install the development dependencies with `python -m pip install -e ".[dev]"`.
3. Make your changes and add or update tests where appropriate.
4. Run the tests and ensure your changes do not break existing functionality.
5. Open a pull request describing what changed and why.

For larger changes, consider raising an issue first so the proposed approach can
be discussed before implementation begins.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
