import { showTooltip } from '../utils/document_utils.js';


export class Hint {
    constructor() {
        this.el = document.querySelector('#hint-btn');

        this.tree = {};
        this.tree_hard = {};
        this.savedBoardState = null;
    }


    init(variables) {
        this.getVariables(variables);
        this.createEvents();
    }


    getVariables(variables) {
        this.cfg = variables.cfg;
        this.game = variables.game;
        this.board = variables.board; // Needed to manipulate the board

        this.tree = variables.dataset.tree;
        this.tree_hard = variables.dataset.tree_hard;
    }


    createEvents() {
        const startEvents = ['mousedown', 'touchstart'];
        const endEvents = ['mouseup', 'touchend', 'mouseleave'];

        // Show tooltip on hover/initial interaction instructions
        // We can keep a click listener for a simple tooltip if they tap quickly, 
        // or just rely on the "Press and hold" mechanic.
        // For this specific "Toggle" feature, press-and-hold is best.
        
        startEvents.forEach(event => {
            this.el.addEventListener(event, (e) => {
                e.preventDefault(); // Prevent ghost clicks on touch
                this.showComparison();
            });
        });

        endEvents.forEach(event => {
            this.el.addEventListener(event, (e) => {
                e.preventDefault();
                this.hideComparison();
            });
        });
        
        // Initial Tooltip (as requested in previous steps)
        setTimeout(() => {
            showTooltip(this.el, "Press and hold to compare your guesses with the optimal path.");
        }, 1000);
    }


    showComparison() {
        if (this.savedBoardState) return; // Already showing

        // 1. Save current board state
        this.savedBoardState = this.board.el.innerHTML;

        // 2. Calculate Optimal Path
        const path = this.getOptimalPath();

        // 3. Render Optimal Path to Board
        this.renderPath(path);
    }


    hideComparison() {
        if (!this.savedBoardState) return;

        // Restore original board
        this.board.el.innerHTML = this.savedBoardState;
        this.savedBoardState = null;
    }


    // Simulates the game using the decision tree to find the solution for the current keyWord
    getOptimalPath() {
        const targetWord = this.game.keyWord;
        const activeTree = this.game.hardmode ? this.tree_hard : this.tree;
        
        let currentNode = activeTree.root;
        const path = [];
        let guesses = 0;

        // Limit to 6 guesses to match board size, though tree guarantees solution
        while (guesses < 6) {
            // 1. Get guess at current node
            // The tree structure is slightly different between root and others? 
            // Based on original solve(), root is an ID, nodes uses that ID.
            const nodeData = activeTree.nodes[currentNode];
            if (!nodeData) break;

            const guess = nodeData.guess.toUpperCase();
            
            // 2. Calculate feedback for this guess against the target
            const feedback = this.calculateFeedback(guess, targetWord);
            
            // 3. Store this step
            path.push({ guess, feedback });

            // 4. Check for win
            if (guess === targetWord) break;

            // 5. Move to next node
            // Key format from solve(): `(currentNode, (f1, f2, f3, f4, f5))`
            const feedbackTuple = `(${feedback.join(', ')})`;
            const key = `(${currentNode}, ${feedbackTuple})`;
            
            if (activeTree.successors[key] !== undefined) {
                currentNode = activeTree.successors[key];
            } else {
                break; // Should not happen if tree is complete
            }
            guesses++;
        }

        return path;
    }


    // Logic replicated from Game.checkGuess to determine colors (2=green, 1=yellow, 0=grey)
    calculateFeedback(guess, target) {
        const wordSize = 5;
        const guessLetters = guess.split('');
        const targetLetters = target.split('');
        
        const feedback = Array(wordSize).fill(0);
        const letterCounts = {};

        // Count frequencies in target
        for (const char of targetLetters) {
            letterCounts[char] = (letterCounts[char] || 0) + 1;
        }

        // Pass 1: Greens
        for (let i = 0; i < wordSize; i++) {
            if (guessLetters[i] === targetLetters[i]) {
                feedback[i] = 2;
                letterCounts[guessLetters[i]]--;
            }
        }

        // Pass 2: Yellows
        for (let i = 0; i < wordSize; i++) {
            if (feedback[i] === 2) continue; // Already handled

            const char = guessLetters[i];
            if (letterCounts[char] > 0) {
                feedback[i] = 1;
                letterCounts[char]--;
            }
        }

        return feedback;
    }


    renderPath(path) {
        // Clear board
        this.board.el.innerHTML = '';

        const maxRows = this.game.numOfGuesses; // 6
        const wordSize = this.game.wordSize; // 5

        for (let i = 0; i < maxRows; i++) {
            const rowDiv = document.createElement('div');
            rowDiv.classList.add('row');

            // If we have a move for this row, render it
            if (i < path.length) {
                const step = path[i];
                const letters = step.guess.split('');
                
                for (let j = 0; j < wordSize; j++) {
                    const cell = document.createElement('div');
                    cell.classList.add('cell');
                    cell.textContent = letters[j];

                    // Apply standard classes
                    if (step.feedback[j] === 2) cell.classList.add('correct');
                    else if (step.feedback[j] === 1) cell.classList.add('present');
                    else cell.classList.add('incorrect');

                    // Add a special style/border to indicate this is "Comparison Mode"
                    // We can reuse existing CSS but maybe add an inline style or opacity
                    cell.style.borderColor = "var(--text-color)"; // Highlight border to show it's special?
                    
                    rowDiv.appendChild(cell);
                }
            } else {
                // Render empty row
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