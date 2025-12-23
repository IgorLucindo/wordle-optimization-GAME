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
        
        this.isAnimating = false;
    }


    init(variables) {
        this.getVariables(variables);
        this.createEvents();
        this.start();
    }


    getVariables(variables) {
        this.cfg = variables.cfg;
        this.dataset = variables.dataset;
        this.board = variables.board;
        this.keyboard = variables.keyboard;
        this.message = variables.message;
        this.hint = variables.hint;

        this.words = variables.dataset.allWords;
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
        this.isAnimating = false;
        this.resetButton.style.display = 'none';
        this.hint.reset();
        this.keyWord = this.selectRandomWord().toUpperCase();
        console.log("Word of the day:", this.keyWord);

        this.keyboard.resetKeyColors();
        this.hardConstraints = { greens: Array(this.wordSize).fill(null), yellows: new Set() };
    }


    selectRandomWord() {
        const keyWords = this.dataset.keyWords;
        return keyWords[Math.floor(Math.random() * keyWords.length)];
    }


    handleKeyPress(key) {
        if (this.gameEnded || this.isAnimating || this.hint.isComparing) return;

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


    handlePhysicalKeyPress(e) {
        const key = e.key.toUpperCase();
        this.handleKeyPress(key);
    }


    changeMode() {
        const diffA = this.diffButton.querySelector('a');
        const diffSVG = this.diffButton.querySelector('svg');
        const svgPathInnerArc = this.diffButton.querySelector('#inner-arc');
        

        this.hardmode = !this.hardmode;

        if (this.hardmode) {
            diffA.innerHTML = `Hard`;
            diffSVG.style.transform = 'scaleX(1)';
            svgPathInnerArc.style.fill = 'var(--red)';
            showTooltip(this.diffButton, "Any revealed hints must be used in subsequent guesses.");
        }
        else {
            diffA.innerHTML = `Regular`;
            diffSVG.style.transform = 'scaleX(-1)';
            svgPathInnerArc.style.fill = 'var(--green)';
            showTooltip(this.diffButton, "Hints are optional and can be ignored for any subsequent guess.");
        }
    }


    checkGuess() {
        const guessString = this.currentGuess.join('');
        const guessLetters = guessString.split('');
        const wordLetters = this.keyWord.split('');
        const currentRowElement = this.board.el.children[this.currentRow];

        // 1. Validation
        if (!this.words.includes(guessString.toLowerCase())) {
            this.board.shakeRow();
            this.message.show('Not in word list');
            return;
        }

        if (this.hardmode && !this.validateHardMode(guessLetters)) {
            this.board.shakeRow();
            return;
        }

        this.isAnimating = true;

        // 2. Calculation
        const letterCounts = {};
        for (const char of wordLetters) letterCounts[char] = (letterCounts[char] || 0) + 1;

        const status = Array(this.wordSize).fill('');
        const feedback = [];

        for (let i = 0; i < this.wordSize; i++) {
            if (guessLetters[i] === wordLetters[i]) {
                status[i] = 'correct';
                letterCounts[guessLetters[i]]--;
            }
        }

        for (let i = 0; i < this.wordSize; i++) {
            if (status[i] === 'correct') {
                feedback.push(2);
                continue;
            }
            if (letterCounts[guessLetters[i]] > 0) {
                status[i] = 'present';
                letterCounts[guessLetters[i]]--;
                feedback.push(1);
            } else {
                status[i] = 'incorrect';
                feedback.push(0);
            }
        }

        // 3. Animation
        const animDuration = 500; 
        const delayBetween = 250; 

        for (let i = 0; i < this.wordSize; i++) {
            setTimeout(() => {
                const cell = currentRowElement.children[i];
                const char = guessLetters[i];

                cell.classList.add('animate-flip');

                setTimeout(() => {
                    if (status[i] === 'correct') {
                        cell.classList.add('correct');
                        this.keyboard.updateKeyColor(char, 'correct');
                    } else if (status[i] === 'present') {
                        cell.classList.add('present');
                        this.keyboard.updateKeyColor(char, 'present');
                    } else {
                        cell.classList.add('incorrect');
                        this.keyboard.updateKeyColor(char, 'incorrect');
                    }
                }, animDuration / 2);

            }, i * delayBetween);
        }

        const totalDelay = (this.wordSize - 1) * delayBetween + animDuration;

        setTimeout(() => {
            this.isAnimating = false;

            for (let i = 0; i < this.wordSize; i++) {
                if (status[i] === 'correct') this.hardConstraints.greens[i] = guessLetters[i];
                else if (status[i] === 'present') this.hardConstraints.yellows.add(guessLetters[i]);
            }

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
        }, totalDelay);
    }


    validateHardMode(guessLetters) {
        const { greens, yellows } = this.hardConstraints;

        for (let i = 0; i < greens.length; i++) {
            if (greens[i] && guessLetters[i] !== greens[i]) {
                this.message.show(`Hard mode: must keep "${greens[i]}" at position ${i + 1}.`);
                this.board.shakeRow();
                return false;
            }
        }

        for (const letter of yellows) {
            if (!guessLetters.includes(letter)) {
                this.message.show(`Hard mode: must include letter "${letter}".`);
                this.board.shakeRow();
                return false;
            }
        }
        return true;
    }
}