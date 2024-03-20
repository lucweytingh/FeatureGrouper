# Feature Grouper

Feature Grouper is a Python tool for grouping textual options based on their semantic similarity. Utilizing TF-IDF vectorization and cosine similarity, it effectively clusters options that share substantial lexical and semantic characteristics. This tool is particularly useful for analyzing and categorizing large sets of text data into coherent groups.

Grouping using TF-IDF is by no means perfect, but it can provide a good starting point for de-duplicating entries.

## Installation

Install these packages using pip:

```bash
pip install pandas scikit-learn
```

## Usage

```bash
python FeatureGrouper.py

# Output:
# Original amount of phenomena: 4079
# With TF-IDF deduplicating the amount of phenomena is reduced to: 2932
```
Adjust the distance threshold to finetune the granularity. 

The GroupName is simply the first option in the group, more sophisticated group naming could be applied.
