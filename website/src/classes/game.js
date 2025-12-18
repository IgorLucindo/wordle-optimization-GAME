import { showTooltip } from '../utils/document_utils.js';


export class Game {
    constructor() {
        this.resetButton = document.getElementById('reset-button');
        this.diffButton = document.getElementById('diff-btn');

        this.wordSize = 5;
        this.numOfGuesses = 6;

        this.words = [];
        this.currentRow = 0;
        this.currentGuess = [];
        this.hardConstraints = { greens: Array(this.wordSize).fill(null), yellows: new Set() };
        this.keyWord = "";
        this.gameEnded = false;
        this.hardmode = false;
    }


    init(variables) {
        this.getVariables(variables);
        this.createEvents();
        this.start();
        this.hint.solve();
    }


    getVariables(variables) {
        this.cfg = variables.cfg;
        this.dataset = variables.dataset;
        this.board = variables.board;
        this.keyboard = variables.keyboard;
        this.message = variables.message;
        this.hint = variables.hint;

        this.words = variables.dataset.allWords
    }


    createEvents() {
        const eventType = this.cfg.touch ? 'touchend' : 'click';

        // Reset button click event
        this.resetButton.addEventListener(eventType, () => {
            this.board.create();
            this.start();
        });

        // Press key event
        document.addEventListener('keydown', (e) => {
            this.handlePhysicalKeyPress(e);
        });

        // Difficulty button click event
        this.diffButton.addEventListener(eventType, () => {
            this.changeMode();
        });
    }


    start() {
        this.currentRow = 0;
        this.currentGuess = [];
        this.gameEnded = false;
        this.resetButton.style.display = 'none';
        this.keyWord = this.selectRandomWord().toUpperCase();
        console.log("Word of the day:", this.keyWord); // For debugging

        this.keyboard.resetKeyColors();
        this.hardConstraints = { greens: Array(this.wordSize).fill(null), yellows: new Set() };
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


    // Changes game mode
    changeMode() {
        const diffA = this.diffButton.querySelector('a');
        const diffSVG = this.diffButton.querySelector('svg');
        const svgPathInnerArc = this.diffButton.querySelector('#inner-arc');
        

        this.hardmode = !this.hardmode;

        if (this.hardmode) {
            diffA.innerHTML = `Hard`;
            diffSVG.style.transform = 'scaleX(1)';
            svgPathInnerArc.style.fill = 'var(--red)'
            showTooltip(this.diffButton, "Any revealed hints must be used in subsequent guesses.")
        }
        else {
            diffA.innerHTML = `Regular`;
            diffSVG.style.transform = 'scaleX(-1)';
            svgPathInnerArc.style.fill = 'var(--green)'
            showTooltip(this.diffButton, "Hints are optional and can be ignored for any subsequent guess.")
        }
    }


    // Checks the current guess against the word of the day and updates the UI
    checkGuess() {
        const guessString = this.currentGuess.join('');
        const guessLetters = guessString.split('');
        const wordLetters = this.keyWord.split('');
        const currentRowElement = this.board.el.children[this.currentRow];

        // 1. Validation: Word exists
        if (!this.words.includes(guessString.toLowerCase())) {
            this.board.shakeRow();
            this.message.show('Not in word list');
            return;
        }

        // 2. Validation: Hard Mode
        if (this.hardmode && !this.validateHardMode(guessLetters)) {
            this.board.shakeRow();
            return;
        }

        // 3. Prepare for Feedback Calculation
        const letterCounts = {};
        for (const char of wordLetters) {
            letterCounts[char] = (letterCounts[char] || 0) + 1;
        }

        const status = Array(this.wordSize).fill('');
        const feedback = [];

        // First Pass: Identify Greens (Correct)
        for (let i = 0; i < this.wordSize; i++) {
            if (guessLetters[i] === wordLetters[i]) {
                status[i] = 'correct';
                letterCounts[guessLetters[i]]--;
            }
        }

        // Second Pass: Identify Yellows/Greys and Update UI
        for (let i = 0; i < this.wordSize; i++) {
            const char = guessLetters[i];
            const cell = currentRowElement.children[i];
            let feedbackValue = 0;

            if (status[i] === 'correct') {
                feedbackValue = 2; // Green
                cell.classList.add('correct');
                this.keyboard.updateKeyColor(char, 'correct');
            } 
            else if (letterCounts[char] > 0) {
                feedbackValue = 1; // Yellow
                status[i] = 'present';
                cell.classList.add('present');
                this.keyboard.updateKeyColor(char, 'present');
                letterCounts[char]--;
            } 
            else {
                feedbackValue = 0; // Grey
                status[i] = 'incorrect';
                cell.classList.add('incorrect');
                this.keyboard.updateKeyColor(char, 'incorrect');
            }

            // Push integer to feedback tuple
            feedback.push(feedbackValue);
        }

        // 4. Update Hard Mode Constraints for next turn
        for (let i = 0; i < this.wordSize; i++) {
            if (status[i] === 'correct') {
                this.hardConstraints.greens[i] = guessLetters[i];
            } else if (status[i] === 'present') {
                this.hardConstraints.yellows.add(guessLetters[i]);
            }
        }

        // 5. Check Game End Conditions
        if (guessString === this.keyWord) {
            this.message.show('You guessed it! 🎉');
            this.gameEnded = true;
        } 
        else if (this.currentRow === this.numOfGuesses - 1) {
            this.message.show(`Game Over! The word was "${this.keyWord}"`);
            this.gameEnded = true;
        } 
        else {
            this.currentRow++;
            this.currentGuess = [];
        }

        // 6. Pass Feedback Tuple to Hint System
        if (this.gameEnded) {
            this.resetButton.style.display = 'block';
        } 
        // Only advance the tree if the user played the suggested word
        else if (this.hint && guessString === this.hint.guess) {
            this.hint.solve(feedback);
        }
    }


    // Validates the current guess against hard mode rules
    validateHardMode(guessLetters) {
        const { greens, yellows } = this.hardConstraints;
        console.log(greens, yellows)

        // Check greens (must match position)
        for (let i = 0; i < greens.length; i++) {
            if (greens[i] && guessLetters[i] !== greens[i]) {
                console.log("green violated")
                this.message.show(`Hard mode: must keep "${greens[i]}" at position ${i + 1}.`);
                this.board.shakeRow();
                return false;
            }
        }

        // Check yellows (must be included somewhere)
        for (const letter of yellows) {
            if (!guessLetters.includes(letter)) {
                console.log("yellow violated")
                this.message.show(`Hard mode: must include letter "${letter}".`);
                this.board.shakeRow();
                return false;
            }
        }

        return true;
    }
}