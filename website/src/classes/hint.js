export class Hint {
    constructor() {
        this.el = document.querySelector('#hint-btn');

        this.words = [];
        this.guess = null;
    }


    init(variables) {
        this.getVariables(variables);
        this.createEvents();
    }


    getVariables(variables) {
        this.cfg = variables.cfg;
        this.message = variables.message;

        this.words = variables.dataset.words;
    }


    createEvents() {
        // Click info button event
        const clickHintBtn = () => {
            this.message.show(`Word hint: <span class="msg-highlight">${this.guess}</span>`);
        };

        // Create events
        if (this.cfg.touch) {
            this.el.addEventListener('touchend', clickHintBtn);
        }
        else {
            this.el.addEventListener('click', clickHintBtn);
        }
    }


    solve(guessResults=null) {
        this.filter_words(guessResults);
        this.setRandomGuess();
    }


    // Handle guess results in order to update possible words
    filter_words(guessResults) {
        if (!guessResults) return;
        
        // Count how many times the letter appears
        const guessResultsFlattened = [
            ...guessResults.correct,
            ...guessResults.present,
            ...guessResults.incorrect
        ];
        const guessLetters = [
            ...new Set(guessResultsFlattened.map(result => result.letter))
        ];
        const letterCount = {};
        guessLetters.forEach(letter => {
            letterCount[letter] = guessResultsFlattened.reduce((count, r) => {
                return count + (r.letter === letter && r.status !== 0 ? 1 : 0);
            }, 0);
        });
        
        // Handle each letter result
        guessResults.correct.forEach(result => {
            this.handleCorrectStatus(result);
        });
        guessResults.incorrect.forEach(result => {
            this.handleIncorrectStatus(result, letterCount);
        });
        guessResults.present.forEach(result => {
            this.handlePresentStatus(result);
        });
    }


    // Letter is in the correct position
    handleCorrectStatus(result) {
        const letter = result.letter;
        const pos = result.pos;

        const wordsWithLetterPos = this.words.filter(word => word[pos] === letter);
        this.words = wordsWithLetterPos;
    }

    
    // Letter is in the word but NOT at this position
    handlePresentStatus(result) {
        const letter = result.letter;
        const pos = result.pos;

        const wordsWithLetterElsewhere = this.words.filter(word =>
            word.includes(letter) && word[pos] !== letter
        );
        this.words = wordsWithLetterElsewhere;
    }


    // Letter appears in the word a defined amount of times
    handleIncorrectStatus(result, letterCount) {
        const letter = result.letter;
        const pos = result.pos;

        const wordsWithNumOfLetters = this.words.filter(word => {
            const letterOccurrences = [...word].filter(c => c === letter).length;
            return letterOccurrences <= letterCount[letter] && word[pos] !== letter;
        });
        this.words = wordsWithNumOfLetters;
    }


    setRandomGuess(){
        this.guess = this.words[Math.floor(Math.random() * this.words.length)];
    }
}