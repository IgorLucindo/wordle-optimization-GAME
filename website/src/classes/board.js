export class Board {
    constructor() {
        this.el = document.getElementById('game-board');
    }


    init(variables) {
        this.getVariables(variables);
        this.create();
    }


    getVariables(variables) {
        this.game = variables.game;
    }


    // Creates the visual game board grid
    create() {
        // Clear existing board
        this.el.innerHTML = '';

        for (let i = 0; i < this.game.numOfGuesses; i++) {
            const row = document.createElement('div');
            row.classList.add('row');

            for (let j = 0; j < this.game.wordSize; j++) {
                const cell = document.createElement('div');
                cell.classList.add('cell');
                row.appendChild(cell);
            }
            this.el.appendChild(row);
        }
    }


    //  Updates the content of the current row on the game board
    update() {
        const currentRowElement = this.el.children[this.game.currentRow];
        
        for (let i = 0; i < this.game.wordSize; i++) {
            const cell = currentRowElement.children[i];
            cell.textContent = this.game.currentGuess[i] || '';
        }
    }


    // Animation for shaking current row
    shakeRow() {
        const currentRowElement = this.el.children[this.game.currentRow];

        for (let i = 0; i < this.game.wordSize; i++) {
            const cell = currentRowElement.children[i];

            cell.classList.remove('shake');
            void cell.offsetWidth;
            cell.classList.add('shake');
        }
    }
}