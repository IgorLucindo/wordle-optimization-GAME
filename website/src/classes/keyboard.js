export class Keyboard {
    constructor(keyboardContainer, keyPressCallback) {
        this.keyboardContainer = keyboardContainer;
        this.keyPressCallback = keyPressCallback; // Callback function for key presses
        this.createKeyboard();
        this.resetKeyColors();
    }

    createKeyboard() {
        const keyboardLayout = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['ENTER', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', 'BACKSPACE']
        ];

        keyboardLayout.forEach(rowKeys => {
            const rowDiv = document.createElement('div');
            rowDiv.classList.add('keyboard-row');
            rowKeys.forEach(keyText => {
                const button = document.createElement('button');
                button.classList.add('key');
                button.textContent = keyText;
                button.id = `key-${keyText.toLowerCase()}`;
                if (keyText === 'ENTER' || keyText === 'BACKSPACE') {
                    button.classList.add('large');
                }
                button.addEventListener('click', () => this.keyPressCallback(keyText));
                rowDiv.appendChild(button);
            });
            this.keyboardContainer.appendChild(rowDiv);
        });
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