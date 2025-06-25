import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from data_preprocessing import clean_text_series


def load_data(train_csv, valid_csv):
    df_train = pd.read_csv(train_csv)
    df_train['split'] = 'train'
    df_valid = pd.read_csv(valid_csv)
    df_valid['split'] = 'valid'
    df = pd.concat([df_train, df_valid], ignore_index=True)
    return df


def plot_sentiment_distribution(df, output_dir):
    plt.figure(figsize=(6,4))
    sns.countplot(x='label', data=df)
    plt.title('Sentiment Distribution')
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'))
    plt.close()


def plot_word_frequency(df, text_col='text', top_n=20, output_dir=None):
    all_words = df[text_col].str.split().explode()
    freq = all_words.value_counts().nlargest(top_n)
    plt.figure(figsize=(8,6))
    sns.barplot(x=freq.values, y=freq.index)
    plt.title(f'Top {top_n} Most Common Words')
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'word_frequency.png'))
    plt.close()


def plot_ticker_frequency(df, top_n=20, output_dir=None):
    all_tickers = df['text'].str.findall(r'\$[A-Za-z]{1,5}').explode()
    freq = all_tickers.value_counts().nlargest(top_n)
    plt.figure(figsize=(8,6))
    sns.barplot(x=freq.values, y=freq.index)
    plt.title(f'Top {top_n} Tickers')
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'ticker_frequency.png'))
    plt.close()


def plot_wordclouds(df, text_col='text', output_dir=None):
    for sentiment in df['label'].unique():
        text_series = df[df['label'] == sentiment][text_col]
        word_freq = text_series.str.split().explode().value_counts()
        if word_freq.empty:
            continue
        wordcloud = WordCloud(width=800, height=400, background_color='white')
        wordcloud.generate_from_frequencies(word_freq)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for Sentiment: {sentiment}')
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'wordcloud_{sentiment}.png'))
        plt.close()


if __name__ == '__main__':
    base = os.path.dirname(__file__)
    train_csv = os.path.join(base, 'dataset', 'sent_train.csv')
    valid_csv = os.path.join(base, 'dataset', 'sent_valid.csv')
    output_dir = os.path.join(base, 'eda_outputs')
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(train_csv, valid_csv)

    df['text_clean'] = clean_text_series(df['text'])

    print("Generating EDA plots...")
    plot_sentiment_distribution(df, output_dir)
    plot_ticker_frequency(df, output_dir=output_dir)
    plot_word_frequency(df, text_col='text_clean', output_dir=output_dir)
    plot_wordclouds(df, text_col='text_clean', output_dir=output_dir)

    print(f"EDA output saved in {output_dir}")
