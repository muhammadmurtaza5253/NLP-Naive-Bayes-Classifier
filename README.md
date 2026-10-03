# NLP Naive Bayes Classifier

Homework 1 for EDS 6352 – Natural Language Processing (University of Houston).

A Naive Bayes sentiment classifier for IMDB movie reviews, implemented from scratch in Python. The project also measures how much different text-preprocessing choices affect accuracy.

## Dataset

[IMDB Movie Ratings Sentiment Analysis](https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis) (Kaggle), included as `movie.csv`. It has 40,000 reviews with a binary label (`0` = negative, `1` = positive).

After dropping duplicates, the data is split as follows:

- **Train:** 1,000 randomly sampled reviews (`random_state=42`)
- **Test:** 400 unseen reviews, balanced at 200 per class

## Approach

- **Classifier:** multinomial Naive Bayes built without ML libraries. It uses class priors, per-class word counts and log-probabilities to avoid underflow. Laplace (add-one) smoothing handles unseen words.
- **Evaluation:** confusion matrix, per-class precision, recall and F1, and overall accuracy, all implemented by hand.
- **Preprocessing:** HTML tag removal and regex cleanup, then spaCy (`en_core_web_sm`) for tokenization, optional lemmatization, optional stop-word removal and optional negation handling. Negation handling tags the words after a negation cue (e.g. "not", "never") until the next punctuation mark.

## Experiments and results

| Scenario | Preprocessing | Accuracy |
|---|---|---|
| 1 | Baseline (none) | 0.792 |
| 2 | + Lemmatization | 0.810 |
| 3 | + Lemmatization + stop-word removal | 0.810 |
| 4 | + Lemmatization + stop-word removal + negation handling | **0.818** |

A fifth experiment runs scenario 4 without Laplace smoothing, to show why smoothing is needed. Without it, any word unseen in a class gives that class zero probability.

## Files

- `Murtaza_HW1.ipynb` – full code, outputs and commentary
- `Murtaza_HW1.pdf` – exported notebook
- `EDS6352 - HW1 1.pdf` – assignment instructions
- `movie.csv` – dataset

## Running it

```bash
pip install pandas numpy spacy matplotlib seaborn kagglehub jupyter
python -m spacy download en_core_web_sm
jupyter notebook Murtaza_HW1.ipynb
```

The notebook's first cell downloads the dataset with `kagglehub`. This is optional if `movie.csv` is already in the folder.
