import re
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, RobertaTokenizer
from sklearn.utils.class_weight import compute_class_weight
import emoji
import random

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split

TICKER_MAP = {}
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")


class TextAugmentation:
    def __init__(self, augment_prob=0.3):
        self.augment_prob = augment_prob
        self.synonyms = {
            'bullish': ['positive', 'optimistic', 'upbeat', 'favorable'],
            'bearish': ['negative', 'pessimistic', 'downbeat', 'unfavorable'],
            'increased': ['rose', 'climbed', 'advanced', 'gained'],
            'decreased': ['fell', 'dropped', 'declined', 'lost'],
            'strong': ['robust', 'solid', 'firm', 'stable'],
            'weak': ['fragile', 'unstable', 'poor', 'soft']
        }

    def synonym_replacement(self, text: str, n: int = 1) -> str:
        words = text.split()
        new_words = words.copy()

        for _ in range(n):
            synonyms_found = []
            for i, word in enumerate(words):
                if word.lower() in self.synonyms:
                    synonyms_found.append((i, word.lower()))

            if synonyms_found:
                idx, word = random.choice(synonyms_found)
                synonym = random.choice(self.synonyms[word])
                new_words[idx] = synonym

        return ' '.join(new_words)

    @staticmethod
    def random_deletion(text: str, p: float = 0.1) -> str:
        words = text.split()
        if len(words) == 1:
            return text

        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)

        return ' '.join(new_words) if new_words else text

    @staticmethod
    def random_swap(text: str, n: int = 1) -> str:
        words = text.split()
        if len(words) < 2:
            return text

        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return ' '.join(words)

    def augment_text(self, text: str) -> str:
        if random.random() > self.augment_prob:
            return text

        aug_type = random.choice(['synonym', 'deletion', 'swap'])

        if aug_type == 'synonym':
            return self.synonym_replacement(text, n=random.randint(1, 2))
        elif aug_type == 'deletion':
            return self.random_deletion(text, p=0.1)
        else:
            return self.random_swap(text, n=1)


def extract_tickers(*series_list, pattern=r"\$[A-Za-z]{1,5}"):
    combined_series = pd.concat(series_list, ignore_index=True).dropna().astype(str)
    return combined_series.str.findall(pattern).explode().dropna().unique().tolist()


def clean_text_series(series: pd.Series, apply_stemming_lemmatization: bool = True) -> pd.Series:
    url_pattern = r"http\S+|www\.\S+"
    mention_pattern = r"@\w+"
    hashtag_pattern = r"#\w+"
    non_alpha_num_pattern = r"[^A-Za-z0-9(),!?\'`]"
    multi_space_pattern = r"\s+"

    series = series.astype(str).apply(emoji.demojize)
    series = series.str.replace(r":([a-zA-Z_]+):", r" \1 ", regex=True)


    series = series.str.replace(mention_pattern, "", regex=True)
    series = series.str.replace(hashtag_pattern, "", regex=True)

    def replace_ticker_in_text(text):
        def _rep(m):
            tk = m.group(0)
            return TICKER_MAP.get(tk, tk.strip('$'))

        return re.sub(r"\$[A-Za-z]{1,5}", _rep, text)

    series = series.apply(replace_ticker_in_text)
    series = series.str.replace(url_pattern, "", regex=True)
    series = series.str.replace(non_alpha_num_pattern, " ", regex=True)
    series = series.str.replace(multi_space_pattern, " ", regex=True).str.strip()

    contractions = {
        "won't": "will not",
        "can't": "can not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'t": " not",
        "'ve": " have",
        "'m": " am"
    }

    def expand_contractions(text):
        for key, value in contractions.items():
            text = text.replace(key, value)
        return text

    series = series.apply(expand_contractions)

    def process_text(text):
        words = text.split()
        processed = []
        for word in words:
            if word.lower() not in stop_words:
                if apply_stemming_lemmatization:
                    stemmed = stemmer.stem(word)
                    lemmatized = lemmatizer.lemmatize(stemmed)
                    processed.append(lemmatized)
                else:
                    processed.append(word)
        return ' '.join(processed)

    series = series.apply(process_text)
    return series


class FinanceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128, augment_prob=0.3,
                 apply_stemming_lemmatization: bool = True):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment_prob = augment_prob
        self.augmenter = TextAugmentation(augment_prob=augment_prob)
        self.apply_stemming_lemmatization = apply_stemming_lemmatization

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        if random.random() < self.augment_prob:
            text = self.augmenter.augment_text(text)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return item


def prepare_dataloaders(
        train_csv_p,
        valid_csv_p,
        model_name='ProsusAI/finbert',
        batch_size=32,
        max_length=128,
        test_size=0.1,
        random_state=42,
        augment_prob=0.3,
        apply_stemming_lemmatization: bool = True,
        use_roberta: bool = False
):
    df_train = pd.read_csv(train_csv_p)
    df_valid = pd.read_csv(valid_csv_p)
    for df, name in [(df_train, 'Train'), (df_valid, 'Valid')]:
        if 'Sentence' not in df.columns or 'Sentiment' not in df.columns:
            raise ValueError(f"{name} CSV must contain 'Sentence' and 'Sentiment' columns")

    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    df_train["label"] = labelencoder.fit_transform(df_train["Sentiment"])
    df_valid["label"] = labelencoder.transform(df_valid["Sentiment"])

    df_train.drop_duplicates(subset=['Sentence'], keep='first', inplace=True)
    df_valid.drop_duplicates(subset=['Sentence'], keep='first', inplace=True)

    tickers = extract_tickers(df_train['Sentence'], df_valid['Sentence'])
    TICKER_MAP.clear()
    TICKER_MAP.update({tk: tk.strip('$') for tk in sorted(tickers)})

    df_train_main, df_test = train_test_split(df_train, test_size=test_size, stratify=df_train['label'],
                                              random_state=random_state)

    x_train_clean = clean_text_series(df_train_main['Sentence'], apply_stemming_lemmatization)
    y_train = df_train_main['label']
    x_valid_clean = clean_text_series(df_valid['Sentence'], apply_stemming_lemmatization)
    y_valid = df_valid['label']
    x_test_clean = clean_text_series(df_test['Sentence'], apply_stemming_lemmatization)
    y_test = df_test['label']

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    tokenizer_cls = RobertaTokenizer if use_roberta else BertTokenizer
    tokenizer = tokenizer_cls.from_pretrained(model_name)
    train_ds = FinanceDataset(x_train_clean, y_train, tokenizer, max_length, augment_prob, apply_stemming_lemmatization)
    valid_ds = FinanceDataset(x_valid_clean, y_valid, tokenizer, max_length, augment_prob, apply_stemming_lemmatization)
    test_ds = FinanceDataset(x_test_clean, y_test, tokenizer, max_length, augment_prob, apply_stemming_lemmatization)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4,
                          pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_dl, valid_dl, test_dl, class_weights


if __name__ == '__main__':
    base = os.path.dirname(__file__)
    train_csv = os.path.join(base, 'dataset', 'sent_train.csv')
    valid_csv = os.path.join(base, 'dataset', 'sent_valid.csv')
    train_loader, valid_loader, test_loader, _ = prepare_dataloaders(train_csv, valid_csv)
    print(
        f"Train batches: {len(train_loader)} | Validation batches: {len(valid_loader)} | Test batches: {len(test_loader)}")