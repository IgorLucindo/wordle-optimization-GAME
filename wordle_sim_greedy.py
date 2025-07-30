# to run locally, enter in terminal: streamlit run game.py

from collections import Counter
from typing import List, Tuple
import math
import time


def load_words_from_file(path: str) -> List[str]:
    with open(path, 'r') as f:
        # t0 = time.process_time()
        word_list = [line.strip().lower() for line in f]
        # t1 = time.process_time()
        # print(f'loading list time: {t1-t0}')
        return word_list

def feedback(guess: str, target: str) -> str:
    word_length = len(target)
    result = ['B'] * word_length
    target_count = Counter(target)

    for i in range(word_length):
        if guess[i] == target[i]:
            result[i] = 'G'
            target_count[guess[i]] -= 1

    for i in range(word_length):
        if result[i] == 'B' and target_count[guess[i]] > 0:
            result[i] = 'Y'
            target_count[guess[i]] -= 1

    return ''.join(result)

def filter_candidates(candidates: List[str], guess: str, result: str) -> List[str]:
    return [word for word in candidates if feedback(guess, word) == result]

def score_words_by_entropy(candidates: List[str]) -> List[Tuple[str, float]]:
    # t0 = time.process_time()
    if not candidates:
        return []

    word_length = len(candidates[0])
    total = len(candidates)
    position_freq = [Counter() for _ in range(word_length)]

    for word in candidates:
        for i, ch in enumerate(word):
            position_freq[i][ch] += 1

    scored = []
    for word in candidates:
        log_prob = 0.0
        for i, ch in enumerate(word):
            freq = position_freq[i][ch]
            if freq == 0:
                continue
            prob = freq / total
            log_prob += math.log2(prob)
        scored.append((word, log_prob))

    scored.sort(key=lambda x: x[1])
    # t1 = time.process_time()
    # print(f'processing time: {t1-t0}')
    return scored

def simulate_game(target: str, word_list: List[str]):
    candidates = word_list.copy()
    history = []
    word_length = len(target)
    turn = 0

    while True:
        turn += 1
        print(f'turn {turn}, candidates size: {len(candidates)}')
        scores = score_words_by_entropy(candidates)
        if not scores:
            history.append((turn, "No guess", "No words left"))
            break

        guess = scores[-1][0]
        print(guess)
        fb = feedback(guess, target)
        history.append((turn, guess, fb))

        if fb == "G" * word_length:
            break

        candidates = filter_candidates(candidates, guess, fb)

    return history



# uploaded_file = st.file_uploader("Upload a dictionary (.txt)", type="txt")
if __name__== "__main__":
    for target_word in ['award','forgo', 'reach']:
        word_list = load_words_from_file('wordle_dict.txt')
        # word_list = load_words_from_file('dict.txt')
        simulate_game(target_word, word_list)