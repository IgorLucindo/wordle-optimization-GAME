import { showTooltip } from '../utils/document_utils.js';


export class Solver {
    constructor() {
        this.el = document.querySelector('#solve-btn');

        this.tree = {};
        this.tree_hard = {};
        this.currentNode = null;
        this.guess = null;

        this.isComparing = false;
        this.savedBoardState = null;
    }


    init(variables) {
        this.getVariables(variables);
        this.createEvents();
        setTimeout(() => {this.showTutorialMessage();}, 1500);
    }


    getVariables(variables) {
        this.cfg = variables.cfg;
        this.game = variables.game;
        this.board = variables.board;
        this.tree = variables.dataset.tree;
        this.tree_hard = variables.dataset.tree_hard;
    }


    createEvents() {
        this.el.addEventListener('click', (e) => {
            e.preventDefault();
            this.toggleComparison();
        });
    }


    showTutorialMessage() {
        showTooltip(this.el, "Click to reveal our solution. Try beating it!", 6);
        this.el.classList.add('tutorial-active');
        this.el.addEventListener('animationend', () => {
            this.el.classList.remove('tutorial-active');
        }, { once: true });
    }


    toggleComparison() {
        if (this.isComparing) {
            this.hideComparison();
            this.el.classList.remove('active');
            showTooltip(this.el, "Solution: Hidden"); 
            this.isComparing = false;
        } else {
            this.showComparison();
            this.el.classList.add('active');
            showTooltip(this.el, "Solution: Visible"); 
            this.isComparing = true;
        }
    }


    // Resets solve state (called when Game restarts)
    reset() {
        this.isComparing = false;
        this.savedBoardState = null;
        this.el.classList.remove('active');
    }


    showComparison() {
        this.savedBoardState = this.board.el.innerHTML;
        const path = this.getOptimalPath();
        this.renderPath(path);

        if (!this.game.gameEnded) {
            this.game.victoryMessage.textContent = "Press Play Again to restart.";
            this.game.victoryContainer.style.display = 'flex';
        }
    }


    hideComparison() {
        if (!this.savedBoardState) return;
        this.board.el.innerHTML = this.savedBoardState;
        this.savedBoardState = null;

        if (!this.game.gameEnded) {
            this.game.victoryContainer.style.display = 'none';
        }
    }


    getOptimalPath() {
        const targetWord = this.game.keyWord;
        const activeTree = this.game.hardmode ? this.tree_hard : this.tree;
        
        let simNode = activeTree.root;
        const path = [];
        let guesses = 0;

        while (guesses < this.game.numOfGuesses) {
            const nodeData = activeTree.nodes[simNode];
            if (!nodeData) break;

            const guess = nodeData.guess.toUpperCase();
            const feedback = this.calculateFeedback(guess, targetWord);
            
            path.push({ guess, feedback });

            if (guess === targetWord) break;

            const feedbackTuple = `(${feedback.join(', ')})`;
            const key = `(${simNode}, ${feedbackTuple})`;
            
            if (activeTree.successors[key] !== undefined) {
                simNode = activeTree.successors[key];
            } else {
                break; 
            }
            guesses++;
        }
        return path;
    }


    calculateFeedback(guess, target) {
        const wordSize = 5;
        const guessLetters = guess.split('');
        const targetLetters = target.split('');
        
        const feedback = Array(wordSize).fill(0);
        const letterCounts = {};

        for (const char of targetLetters) letterCounts[char] = (letterCounts[char] || 0) + 1;

        for (let i = 0; i < wordSize; i++) {
            if (guessLetters[i] === targetLetters[i]) {
                feedback[i] = 2;
                letterCounts[guessLetters[i]]--;
            }
        }

        for (let i = 0; i < wordSize; i++) {
            if (feedback[i] === 2) continue;
            if (letterCounts[guessLetters[i]] > 0) {
                feedback[i] = 1;
                letterCounts[guessLetters[i]]--;
            }
        }
        return feedback;
    }


    renderPath(path) {
        this.board.el.innerHTML = '';
        const wordSize = this.game.wordSize;
        const animationDuration = 250; // duration of the flip (.fast)
        const rowDelay = 470; // Time between rows (flip + reading time)

        // 1. Create the grid structure (empty initially)
        const rows = [];
        for (let i = 0; i < this.game.numOfGuesses; i++) {
            const rowDiv = document.createElement('div');
            rowDiv.classList.add('row');
            
            const cells = [];
            for (let j = 0; j < wordSize; j++) {
                const cell = document.createElement('div');
                cell.classList.add('cell');
                rowDiv.appendChild(cell);
                cells.push(cell);
            }
            this.board.el.appendChild(rowDiv);
            rows.push({ div: rowDiv, cells: cells });
        }

        // 2. Animate step-by-step
        path.forEach((step, i) => {
            setTimeout(() => {
                const row = rows[i];
                
                // Fill text and trigger animation for this row
                for (let j = 0; j < wordSize; j++) {
                    const cell = row.cells[j];
                    cell.textContent = step.guess[j];
                    cell.style.borderColor = "var(--text-color)";

                    // Stagger flip slightly for visual effect within the row
                    setTimeout(() => {
                        cell.classList.add('animate-flip', 'fast');

                        // Reveal color halfway
                        setTimeout(() => {
                            if (step.feedback[j] === 2) cell.classList.add('correct');
                            else if (step.feedback[j] === 1) cell.classList.add('present');
                            else cell.classList.add('incorrect');
                        }, animationDuration / 2);

                    }, j * 50); // Small wave inside the row
                }

            }, i * rowDelay); // Wait for previous rows to finish
        });
    }
}