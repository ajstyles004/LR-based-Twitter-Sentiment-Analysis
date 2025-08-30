import re

__all__ = ['clean_tweet']

_url_re = re.compile(r'http\S+')
_mention_re = re.compile(r'@\w+')
_hashtag_symbol_re = re.compile(r'#')
_non_letter_re = re.compile(r'[^a-z\s]')

def clean_tweet(tweet: str) -> str:
    # Basic tweet cleaner:
    # - lowercases
    # - removes URLs, @mentions
    # - keeps hashtag words (drops only the '#' symbol)
    # - strips non-letters (emojis, digits, punctuation) -> keep spaces + letters
    if not isinstance(tweet, str):
        tweet = str(tweet)
    t = tweet.lower()
    t = _url_re.sub('', t)
    t = _mention_re.sub('', t)
    t = _hashtag_symbol_re.sub('', t)  # keep hashtag word, remove '#'
    t = _non_letter_re.sub(' ', t)     # keep only letters + spaces
    t = ' '.join(t.split())
    return t.strip()
