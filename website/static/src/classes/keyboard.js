export class Keyboard {
    constructor() {
        this.el = document.getElementById('keyboard');
    }


    init(variables) {
        this.getVariables(variables);
        this.create();
        this.resetKeyColors();
    }


    getVariables(variables) {
        this.cfg = variables.cfg;
        this.game = variables.game;
    }


    create() {
        const keyboardLayout = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['ENTER', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', 'BACKSPACE']
        ];

        keyboardLayout.forEach((rowKeys) => {
            const rowDiv = document.createElement('div');
            rowDiv.classList.add('keyboard-row');

            rowKeys.forEach((keyText) => {
                const button = this.createKey(keyText);

                // Click button event
                const clickButton = () => this.game.handleKeyPress(keyText);
                
                // Create events
                if (!this.cfg.touch) button.addEventListener('click', clickButton);
                else button.addEventListener('touchend', clickButton);

                // Append button
                rowDiv.appendChild(button);
            });
            this.el.appendChild(rowDiv);
        });
    }


    createKey(keyText) {
        const button = document.createElement('button');
        button.classList.add('key');
        button.textContent = keyText;
        button.id = `key-${keyText.toLowerCase()}`;

        if (keyText === 'ENTER') button.classList.add('large');
        else if (keyText === 'BACKSPACE') {
            button.innerHTML = `
                <svg aria-hidden="true" xmlns="http://www.w3.org/2000/svg" height="20" viewBox="0 0 24 24" width="20" data-testid="icon-backspace">
                    <path fill="#ffffff" d="M22 3H7c-.69 0-1.23.35-1.59.88L0 12l5.41 8.11c.36.53.9.89 1.59.89h15c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H7.07L2.4 12l4.66-7H22v14zm-11.59-2L14 13.41 17.59 17 19 15.59 15.41 12 19 8.41 17.59 7 14 10.59 10.41 7 9 8.41 12.59 12 9 15.59z"></path>
                </svg>
            `;
        }

        return button;
    }


    /**
     * Updates the color of a specific key on the keyboard.
     * @param {string} key The key character (e.g., 'A', 'ENTER').
     * @param {string} status The status to apply ('correct', 'present', 'absent').
     */
    updateKeyColor(key, status) {
        const keyButton = document.getElementById(`key-${key.toLowerCase()}`);
        if (keyButton) {
            // Ensure 'correct' takes precedence over 'present', and 'present' over 'absent'
            if (status === 'correct') {
                keyButton.classList.remove('present', 'absent');
                keyButton.classList.add('correct');
            } else if (status === 'present' && !keyButton.classList.contains('correct')) {
                keyButton.classList.remove('absent');
                keyButton.classList.add('present');
            } else if (status === 'absent' && !keyButton.classList.contains('correct') && !keyButton.classList.contains('present')) {
                keyButton.classList.add('absent');
            }
        }
    }


    /**
     * Resets all keyboard key colors to their default state.
     */
    resetKeyColors() {
        document.querySelectorAll('.key').forEach(key => {
            key.classList.remove('correct', 'present', 'absent');
        });
    }
}