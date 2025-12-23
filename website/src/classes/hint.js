import { showTooltip } from '../utils/document_utils.js';


export class Hint {
    constructor() {
        this.el = document.querySelector('#hint-btn');

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

        setTimeout(() => {
            showTooltip(this.el, "Click to toggle comparison with the optimal strategy.");
        }, 1000);
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


    toggleComparison() {
        if (this.isComparing) {
            this.hideComparison();
            this.el.classList.remove('active');
            this.isComparing = false;
        } else {
            this.showComparison();
            this.el.classList.add('active');
            this.isComparing = true;
        }
    }


    showComparison() {
        this.savedBoardState = this.board.el.innerHTML;
        const path = this.getOptimalPath();
        this.renderPath(path);
    }


    hideComparison() {
        if (!this.savedBoardState) return;
        this.board.el.innerHTML = this.savedBoardState;
        this.savedBoardState = null;
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
        const animationDuration = 250; // .fast duration

        for (let i = 0; i < this.game.numOfGuesses; i++) {
            const rowDiv = document.createElement('div');
            rowDiv.classList.add('row');

            if (i < path.length) {
                const step = path[i];
                for (let j = 0; j < wordSize; j++) {
                    const cell = document.createElement('div');
                    cell.classList.add('cell');
                    cell.textContent = step.guess[j];
                    
                    // Add special border style for comparison mode
                    cell.style.borderColor = "var(--text-color)";

                    // Animate flip quickly
                    // We stagger the animations slightly for a "wave" effect
                    const delay = (i * wordSize + j) * 50; // 50ms per cell

                    setTimeout(() => {
                        cell.classList.add('animate-flip', 'fast');
                        
                        // Reveal color halfway through flip
                        setTimeout(() => {
                            if (step.feedback[j] === 2) cell.classList.add('correct');
                            else if (step.feedback[j] === 1) cell.classList.add('present');
                            else cell.classList.add('incorrect');
                        }, animationDuration / 2);
                    }, delay);
                    
                    rowDiv.appendChild(cell);
                }
            } else {
                for (let j = 0; j < wordSize; j++) {
                    const cell = document.createElement('div');
                    cell.classList.add('cell');
                    rowDiv.appendChild(cell);
                }
            }
            this.board.el.appendChild(rowDiv);
        }
    }
}