"""Trump's sentiment towards the most-cited people, over time.

Builds on `analyse_sentiment_kaggle.py`: it downloads the same Kaggle
`austinreese/trump-tweets` dataset, but instead of summarising overall
sentiment it asks a sharper question:

    *Of all the people Trump tweets about, who does he mention most --
     and how warm or hostile is he towards each of them, year by year?*

For every tweet we score its sentiment with TextBlob (polarity in [-1, 1]),
attribute the tweet to whichever tracked person it names, then average the
polarity per person per year and draw one line per person.

Run from the repo root:  python day-2/analyse_sentiment_people_over_time.py
"""

import os
import re
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob

HERE = Path(__file__).parent

# How many of the most-cited people to chart, and the minimum number of
# tweets a person needs in a given year for that year's point to be drawn
# (keeps single-tweet years from producing wild spikes).
TOP_N = 6
MIN_TWEETS_PER_YEAR = 5

# People Trump regularly tweets about. Each entry maps a display name to the
# aliases/nicknames we match (whole-word, case-insensitive). Ranking by how
# many tweets mention each one keeps the chart data-driven, not hand-picked.
PEOPLE = {
    "Obama": ["obama"],
    "Hillary Clinton": ["hillary", "crooked hillary"],
    "Joe Biden": ["biden", "sleepy joe"],
    "Robert Mueller": ["mueller"],
    "Nancy Pelosi": ["pelosi"],
    "Jeb Bush": ["jeb"],
    "Marco Rubio": ["rubio"],
    "Adam Schiff": ["schiff"],
    "Bernie Sanders": ["bernie sanders", "bernie"],
    "James Comey": ["comey"],
    "Mitt Romney": ["romney"],
    "Ted Cruz": ["ted cruz", "lyin.? ted"],
    "Vladimir Putin": ["putin"],
    "Chuck Schumer": ["schumer"],
    "John McCain": ["mccain", "mc cain"],
    "Elizabeth Warren": ["elizabeth warren", "pocahontas"],
    "Xi Jinping": ["xi jinping", "president xi"],
    "Kim Jong Un": ["kim jong"],
    "Megyn Kelly": ["megyn"],
}


def load_kaggle_data(data):
    print(f"Downloading {data} dataset...")
    path = kagglehub.dataset_download(f"austinreese/{data}")
    print(f"Dataset downloaded to: {path}")

    # Prefer a CSV that has both the tweet text and a date column.
    csv_file = None
    for root, _dirs, files in os.walk(path):
        for file in files:
            if not file.endswith(".csv"):
                continue
            candidate = os.path.join(root, file)
            cols = pd.read_csv(candidate, nrows=0).columns
            if "content" in cols and "date" in cols:
                csv_file = candidate
                break
        if csv_file:
            break

    if csv_file is None:
        raise FileNotFoundError("No suitable CSV (content + date) in the dataset")

    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file, usecols=["content", "date"])
    print(f"\nDataset shape: {df.shape}")
    return df


def compile_patterns(people):
    """Pre-compile one whole-word regex per person from their aliases."""
    return {
        name: re.compile(r"\b(?:" + "|".join(aliases) + r")\b", re.IGNORECASE)
        for name, aliases in people.items()
    }


def polarity(text):
    """TextBlob polarity in [-1, 1]; 0.0 for empty/non-string content."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    return TextBlob(text).sentiment.polarity


def build_person_table(df, patterns):
    """Long table of (person, year, polarity) -- one row per person mentioned
    in each tweet, so a tweet naming two people counts towards both."""
    rows = []
    for content, year, pol in zip(df["content"], df["year"], df["polarity"]):
        if not isinstance(content, str):
            continue
        for name, pattern in patterns.items():
            if pattern.search(content):
                rows.append((name, year, pol))
    return pd.DataFrame(rows, columns=["person", "year", "polarity"])


if __name__ == "__main__":
    tweets_df = load_kaggle_data("trump-tweets")

    # Parse dates and score every tweet once.
    tweets_df["date"] = pd.to_datetime(tweets_df["date"], errors="coerce")
    tweets_df = tweets_df.dropna(subset=["date"])
    tweets_df["year"] = tweets_df["date"].dt.year

    print(f"\nScoring sentiment for {len(tweets_df):,} tweets with TextBlob...")
    tweets_df["polarity"] = tweets_df["content"].apply(polarity)

    patterns = compile_patterns(PEOPLE)
    person_df = build_person_table(tweets_df, patterns)

    # Rank people by how many tweets mention them, keep the top N.
    mention_counts = person_df["person"].value_counts()
    top_people = mention_counts.head(TOP_N).index.tolist()

    print("\n" + "=" * 60)
    print(f"MOST-CITED PEOPLE (top {TOP_N})")
    print("=" * 60)
    for name in top_people:
        subset = person_df[person_df["person"] == name]
        print(f"  {name:<18} {len(subset):>5,} tweets   "
              f"avg sentiment {subset['polarity'].mean():+.3f}")

    # Average polarity per person per year, dropping thin years.
    top_df = person_df[person_df["person"].isin(top_people)]
    grouped = top_df.groupby(["person", "year"])
    yearly = grouped["polarity"].mean()
    counts = grouped["polarity"].size()
    yearly = yearly[counts >= MIN_TWEETS_PER_YEAR].reset_index()

    # Plot: one line per person, sentiment over time.
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(14, 8))

    for name in top_people:
        series = yearly[yearly["person"] == name].sort_values("year")
        if series.empty:
            continue
        ax.plot(series["year"], series["polarity"],
                marker="o", linewidth=2, label=name)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, label="Neutral")
    ax.set_title("Donald Trump's tweet sentiment towards the most-cited people, over time",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average sentiment polarity  (-1 hostile … +1 warm)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Person", loc="best", ncol=2)
    plt.tight_layout()

    output_png = HERE / "sentiment_people_over_time.png"
    plt.savefig(str(output_png), dpi=200, bbox_inches="tight")
    print(f"\nVisualization saved as '{output_png}'")
    plt.show()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
