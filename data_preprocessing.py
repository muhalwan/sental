import re
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.utils.class_weight import compute_class_weight
import nltk
import emoji
import random
from typing import List, Optional

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, StratifiedKFold

TICKER_MAP = {}
stop_words = set(stopwords.words('english'))

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
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        return ' '.join(new_words) if new_words else text
    
    def random_swap(self, text: str, n: int = 1) -> str:
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


def clean_text_series(series: pd.Series) -> pd.Series:
    series = series.astype(str).apply(emoji.demojize)
    series = series.str.replace(r":([a-zA-Z_]+):", r" \1 ", regex=True)
    series = series.str.replace(r"#\w+", "", regex=True)
    series = series.str.replace(r"http\S+|www\.\S+", "", regex=True)
    series = series.str.replace(r"@\w+", "", regex=True)

    def replace_ticker_in_text(text):
        def _rep(m):
            tk = m.group(0)
            return TICKER_MAP.get(tk, tk.strip('$'))

        return re.sub(r"\$[A-Za-z]{1,5}", _rep, text)

    series = series.apply(replace_ticker_in_text)

    series = series.str.replace(r"[^A-Za-z0-9(),!?\'`]", " ", regex=True)
    series = series.str.replace(r"\s+", " ", regex=True).str.strip()

    def remove_stopwords(text):
        return ' '.join([tok for tok in text.split() if tok.lower() not in stop_words])

    series = series.apply(remove_stopwords)
    return series


class FinanceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128, augment_prob=0.3):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment_prob = augment_prob
        self.augmenter = TextAugmentation(augment_prob=augment_prob)

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
        train_csv,
        valid_csv,
        model_name='ProsusAI/finbert',
        batch_size=32,
        max_length=128,
        test_size=0.1,
        random_state=42,
        augment_prob=0.3
):
    df_train = pd.read_csv(train_csv)
    df_valid = pd.read_csv(valid_csv)
    for df, name in [(df_train, 'Train'), (df_valid, 'Valid')]:
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"{name} CSV must contain 'text' and 'label' columns")

    tickers = extract_tickers(df_train['text'], df_valid['text'])
    TICKER_MAP.clear()
    TICKER_MAP.update({tk: tk.strip('$') for tk in sorted(tickers)})

    df_train_main, df_test = train_test_split(df_train, test_size=test_size, stratify=df_train['label'],
                                              random_state=random_state)

    X_train_clean = clean_text_series(df_train_main['text'])
    y_train = df_train_main['label']
    X_valid_clean = clean_text_series(df_valid['text'])
    y_valid = df_valid['label']
    X_test_clean = clean_text_series(df_test['text'])
    y_test = df_test['label']

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_ds = FinanceDataset(X_train_clean, y_train, tokenizer, max_length, augment_prob)
    valid_ds = FinanceDataset(X_valid_clean, y_valid, tokenizer, max_length, augment_prob)
    test_ds = FinanceDataset(X_test_clean, y_test, tokenizer, max_length, augment_prob)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    return train_dl, valid_dl, test_dl, class_weights


if __name__ == '__main__':
    base = os.path.dirname(__file__)
    train_csv = os.path.join(base, 'dataset', 'sent_train.csv')
    valid_csv = os.path.join(base, 'dataset', 'sent_valid.csv')
    train_loader, valid_loader, test_loader, _ = prepare_dataloaders(train_csv, valid_csv)
    print(
        f"Train batches: {len(train_loader)} | Validation batches: {len(valid_loader)} | Test batches: {len(test_loader)}")