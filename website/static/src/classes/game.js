export class Game {
    constructor() {
        this.words = [];
        this.resetButton = document.getElementById('reset-button');
        this.messageElement = document.getElementById('message');

        this.wordSize = 5;
        this.numOfGuesses = 6;

        this.currentRow = 0;
        this.currentGuess = [];
        this.wordOfTheDay = '';
        this.gameEnded = false;
        this.results = [];
        this.messageTimeout = null;
    }


    init(variables) {
        this.getVariables(variables);
        this.createEvents();
        this.start();
    }


    getVariables(variables) {
        this.dataset = variables.dataset;
        this.board = variables.board;
        this.keyboard = variables.keyboard;

        this.words = variables.dataset.words
    }


    createEvents() {
        // Reset button click event
        this.resetButton.addEventListener('click', () => {
            this.board.create();
            this.start()
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
        this.messageElement.textContent = '';
        this.resetButton.style.display = 'none';
        this.wordOfTheDay = this.selectRandomWord().toUpperCase();
        console.log("Word of the day:", this.wordOfTheDay); // For debugging

        this.keyboard.resetKeyColors();
    }


    // Selects a random word from the provided word list
    selectRandomWord() {
        const solutionWords = this.dataset.solutionWords;
        return solutionWords[Math.floor(Math.random() * solutionWords.length)];
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
        const wordLetters = this.wordOfTheDay.split('');
        const guessString = this.currentGuess.join('');
        const guessLetters = guessString.split('');
        const currentRowElement = this.board.el.children[this.currentRow];

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
            this.results.push({ letter: guessLetters[i], pos: i, status: 2 });
        }

        // Second pass: Mark 'present' (yellow) and 'absent' (grey) letters
        for (let i = 0; i < this.wordSize; i++) {
            const cell = currentRowElement.children[i];

            // Only process if not already marked 'correct'
            if (cell.classList.contains('correct')) continue;

            if (wordLetterCounts[guessLetters[i]] > 0) {
                cell.classList.add('present');
                this.keyboard.updateKeyColor(guessLetters[i], 'present');
                wordLetterCounts[guessLetters[i]]--; // Consume this letter
                this.results.push({ letter: guessLetters[i], pos: i, status: 1 });
            }
            else {
                cell.classList.add('absent');
                this.keyboard.updateKeyColor(guessLetters[i], 'absent');
                this.results.push({ letter: guessLetters[i], pos: i, status: 0 });
            }
        }

        if (guessString === this.wordOfTheDay) {
            this.showMessage('You guessed it! 🎉');
            this.gameEnded = true;
            this.resetButton.style.display = 'block';
        }
        else if (this.currentRow === this.numOfGuesses - 1) {
            this.showMessage(`Game Over! The word was "${this.wordOfTheDay}"`);
            this.gameEnded = true;
            this.resetButton.style.display = 'block';
        }
        else {
            this.currentRow++;
            this.currentGuess = [];
        }
    }

    
    // Displays a message to the user
    showMessage(msg) {
        this.messageElement.innerHTML = msg;

        if (this.messageTimeout) clearTimeout(this.messageTimeout);
        this.messageTimeout = setTimeout(() => {
            this.messageElement.textContent = '';
        }, 3000);
    }
}