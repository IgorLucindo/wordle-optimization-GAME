import { Keyboard } from './keyboard.js';


export class Game {
    constructor(words) {
        this.words = words;
        this.gameBoardElement = document.getElementById('game-board');
        this.messageElement = document.getElementById('reset-button');
        this.resetButton = document.getElementById('message');

        this.wordSize = 5;
        this.numOfGuesses = 6;

        this.currentRow = 0;
        this.currentGuess = [];
        this.wordOfTheDay = '';
        this.gameEnded = false;

        this.keyboard = new Keyboard(document.getElementById('keyboard'), this.handleKeyPress.bind(this));

        this.resetButton.addEventListener('click', () => this.start());
        document.addEventListener('keydown', (event) => this.handlePhysicalKeyPress(event));
    }


    start() {
        this.currentRow = 0;
        this.currentGuess = [];
        this.gameEnded = false;
        this.messageElement.textContent = '';
        this.resetButton.style.display = 'none';
        this.wordOfTheDay = this.selectRandomWord().toUpperCase();
        console.log("Word of the day:", this.wordOfTheDay); // For debugging

        this.gameBoardElement.innerHTML = ''; // Clear existing board
        this.createGameBoard();
        this.keyboard.resetKeyColors();
    }


    /**
     * Selects a random word from the provided word list.
     * @returns {string} The randomly selected word.
     */
    selectRandomWord() {
        return this.words[Math.floor(Math.random() * this.words.length)];
    }


    /**
     * Creates the visual game board grid.
     */
    createGameBoard() {
        for (let i = 0; i < this.numOfGuesses; i++) {
            const row = document.createElement('div');
            row.classList.add('row');
            for (let j = 0; j < this.wordSize; j++) {
                const cell = document.createElement('div');
                cell.classList.add('cell');
                row.appendChild(cell);
            }
            this.gameBoardElement.appendChild(row);
        }
    }


    /**
     * Handles a key press event, whether from the on-screen keyboard or physical keyboard.
     * @param {string} key The character or command of the pressed key.
     */
    handleKeyPress(key) {
        if (this.gameEnded) return;

        if (key === 'BACKSPACE') {
            this.currentGuess.pop();
        } else if (key === 'ENTER') {
            if (this.currentGuess.length === this.wordSize) {
                this.checkGuess();
            } else {
                this.showMessage('Not enough letters!');
            }
        } else if (key.length === 1 && key >= 'A' && key <= 'Z' && this.currentGuess.length < this.wordSize) {
            this.currentGuess.push(key);
        }
        this.updateGameBoard();
    }


    /**
     * Handles physical keyboard input.
     * @param {KeyboardEvent} event The keyboard event object.
     */
    handlePhysicalKeyPress(event) {
        const key = event.key.toUpperCase();
        this.handleKeyPress(key);
    }


    /**
     * Updates the content of the current row on the game board.
     */
    updateGameBoard() {
        const currentRowElement = this.gameBoardElement.children[this.currentRow];
        for (let i = 0; i < this.wordSize; i++) {
            const cell = currentRowElement.children[i];
            cell.textContent = this.currentGuess[i] || '';
        }
    }


    /**
     * Checks the current guess against the word of the day and updates the UI.
     */
    checkGuess() {
        const guessString = this.currentGuess.join('');
        const wordLetters = this.wordOfTheDay.split('');
        const guessLetters = guessString.split('');
        const currentRowElement = this.gameBoardElement.children[this.currentRow];

        // Create a mutable copy of letter counts for the word to handle duplicates
        const wordLetterCounts = {};
        for (const char of wordLetters) {
            wordLetterCounts[char] = (wordLetterCounts[char] || 0) + 1;
        }

        // First pass: Mark 'correct' (green) letters and consume counts
        for (let i = 0; i < this.wordSize; i++) {
            const cell = currentRowElement.children[i];
            if (guessLetters[i] === wordLetters[i]) {
                cell.classList.add('correct');
                this.keyboard.updateKeyColor(guessLetters[i], 'correct');
                wordLetterCounts[guessLetters[i]]--; // Consume this letter
            }
        }

        // Second pass: Mark 'present' (yellow) and 'absent' (grey) letters
        for (let i = 0; i < this.wordSize; i++) {
            const cell = currentRowElement.children[i];
            // Only process if not already marked 'correct'
            if (!cell.classList.contains('correct')) {
                if (wordLetterCounts[guessLetters[i]] > 0) {
                    cell.classList.add('present');
                    this.keyboard.updateKeyColor(guessLetters[i], 'present');
                    wordLetterCounts[guessLetters[i]]--; // Consume this letter
                } else {
                    cell.classList.add('absent');
                    this.keyboard.updateKeyColor(guessLetters[i], 'absent');
                }
            }
        }

        if (guessString === this.wordOfTheDay) {
            this.showMessage('You guessed it! 🎉');
            this.gameEnded = true;
            this.resetButton.style.display = 'block';
        } else if (this.currentRow === this.numOfGuesses - 1) {
            this.showMessage(`Game Over! The word was "${this.wordOfTheDay}"`);
            this.gameEnded = true;
            this.resetButton.style.display = 'block';
        } else {
            this.currentRow++;
            this.currentGuess = [];
        }
    }

    
    /**
     * Displays a message to the user.
     * @param {string} msg The message to display.
     */
    showMessage(msg) {
        this.messageElement.textContent = msg;
    }
}