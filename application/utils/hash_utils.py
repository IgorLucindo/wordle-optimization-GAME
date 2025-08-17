import hashlib


def hash_word_set(word_set):
    return hashlib.sha1("".join(sorted(word_set)).encode()).hexdigest()