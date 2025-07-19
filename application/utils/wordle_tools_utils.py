def update_possible_words_dict(possible_words_dict, game_results):
    """
    Handle game results in order to update possible words dict
    """
    for result in game_results:
        for i in range(len(result)):
            if result[i] == 2:
                pass
                # code for correct letter position
            elif result[i] == 1:
                pass
                # code for incorrect letter position
            else:
                pass
                # code for incorrect letter

    return possible_words_dict
    

def get_random_word():
    return "hello"


def update_game_results(game_results, selected_word, word_guess):
    word_result = []

    for i in range(len(selected_word)):
        if selected_word[i] == word_guess[i]:
            word_result.append(2)
        elif selected_word[i] in word_guess:
            word_result.append(1)
        else:
            word_result.append(0)

    game_results.append(word_result)

    return game_results