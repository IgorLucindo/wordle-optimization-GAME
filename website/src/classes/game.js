export class Game {
    constructor() {
        this.resetButton = document.getElementById('reset-button');

        this.wordSize = 5;
        this.numOfGuesses = 6;

        this.words = [];
        this.currentRow = 0;
        this.currentGuess = [];
        this.keyWord = "";
        this.gameEnded = false;
    }


    init(variables) {
        this.getVariables(variables);
        this.createEvents();
        this.start();
        this.hint.solve();
    }


    getVariables(variables) {
        this.dataset = variables.dataset;
        this.board = variables.board;
        this.keyboard = variables.keyboard;
        this.message = variables.message;
        this.hint = variables.hint;

        this.words = variables.dataset.allWords
    }


    createEvents() {
        // Reset button click event
        this.resetButton.addEventListener('click', () => {
            this.board.create();
            this.start();
        });

        // Press key event
        document.addEventListener('keydown', (e) => {
            this.handlePhysicalKeyPress(e);
        });
    }


    start() {
        this.currentRow = 0;
        this.currentGuess = [];
        this.gameEnded = false;
        this.resetButton.style.display = 'none';
        this.keyWord = this.selectRandomWord().toUpperCase();
        this.keyWord = "ACUTE"
        console.log("Word of the day:", this.keyWord); // For debugging

        this.keyboard.resetKeyColors();
    }


    // Selects a random word from the provided word list
    selectRandomWord() {
        const keyWords = this.dataset.keyWords;
        return keyWords[Math.floor(Math.random() * keyWords.length)];
    }


    // Handles a key press event, whether from the on-screen keyboard or physical keyboard
    handleKeyPress(key) {
        if (this.gameEnded) return;

        if (key === 'BACKSPACE') {
            this.currentGuess.pop();
        }
        else if (key === 'ENTER') {
            if (this.currentGuess.length === this.wordSize) this.checkGuess();
            else this.board.shakeRow();
        }
        else if (key.length === 1 && key >= 'A' && key <= 'Z' && this.currentGuess.length < this.wordSize) {
            this.currentGuess.push(key);
        }
        this.board.update();
    }


    // Handles physical keyboard input
    handlePhysicalKeyPress(e) {
        const key = e.key.toUpperCase();
        this.handleKeyPress(key);
    }


    // Checks the current guess against the word of the day and updates the UI
    checkGuess() {
        const wordLetters = this.keyWord.split('');
        const guessString = this.currentGuess.join('');
        const guessLetters = guessString.split('');
        const currentRowElement = this.board.el.children[this.currentRow];
        let feedback = "";

        // Skip if it is not a word
        if (!this.words.includes(guessString.toLowerCase())) {
            this.board.shakeRow();
            return;
        }

        // Create a mutable copy of letter counts for the word to handle duplicates
        const wordLetterCounts = {};
        for (const char of wordLetters) {
            wordLetterCounts[char] = (wordLetterCounts[char] || 0) + 1;
        }

        // First pass: Mark 'correct' (green) letters and consume counts
        for (let i = 0; i < this.wordSize; i++) {
            if (guessLetters[i] !== wordLetters[i]) continue;

            const cell = currentRowElement.children[i];
            
            cell.classList.add('correct');
            this.keyboard.updateKeyColor(guessLetters[i], 'correct');
            wordLetterCounts[guessLetters[i]]--; // Consume this letter
            feedback += "G";
        }

        // Second pass: Mark 'present' (yellow) and 'incorrect' (grey) letters
        for (let i = 0; i < this.wordSize; i++) {
            const cell = currentRowElement.children[i];

            // Only process if not already marked 'correct'
            if (cell.classList.contains('correct')) continue;

            if (wordLetterCounts[guessLetters[i]] > 0) {
                cell.classList.add('present');
                this.keyboard.updateKeyColor(guessLetters[i], 'present');
                wordLetterCounts[guessLetters[i]]--; // Consume this letter
                feedback += "Y";
            }
            else {
                cell.classList.add('incorrect');
                this.keyboard.updateKeyColor(guessLetters[i], 'incorrect');
                feedback += "B";
            }
        }

        // Handle game over
        if (guessString === this.keyWord) {
            this.message.show('You guessed it! 🎉');
            this.gameEnded = true;
            this.resetButton.style.display = 'block';
        }
        else if (this.currentRow === this.numOfGuesses - 1) {
            this.message.show(`Game Over! The word was "${this.keyWord}"`);
            this.gameEnded = true;
            this.resetButton.style.display = 'block';
        }
        else {
            this.currentRow++;
            this.currentGuess = [];
        }

        // Update guess result
        this.hint.solve(feedback);
    }
}