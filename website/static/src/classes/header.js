import { showOverlay, hideOverlay } from '../utils/document_utils.js';
import { callPythonScript } from '../utils/api_utils.js';


export class Header {
    constructor() {
        this.el = document.querySelector('header');
        this.footerEl = document.querySelector('footer');
    }


    init(variables) {
        this.getVariables(variables);
        this.createEvents();
    }


    getVariables(variables) {
        this.cfg = variables.cfg;
        this.game = variables.game;
    }


    createEvents() {
        const hintBtn = this.el.querySelector('#hint-btn');
        const infoBtn = this.el.querySelector('#info-btn');
        const footer = document.querySelector('footer');

        // Hint button
        const clickHintBtn = async () => {
            const data = await callPythonScript('/get-guess');
            const wordGuess = data.wordGuess.toUpperCase();
            this.game.showMessage(`Word hint: <span class="msg-highlight">${wordGuess}</span>`);
        };

        // Click info button event
        const clickInfoBtn = () => {
            footer.classList.add('visible');
            showOverlay();
        };

        // Click window event
        const clickWindow = (e) => {
            if (!e.target.closest('#info-btn') && !e.target.closest('footer')) {
                footer.classList.remove('visible');
                hideOverlay();
            }
        };

        // Create events
        if (this.cfg.touch) {
            hintBtn.addEventListener('touchend', clickHintBtn);
            infoBtn.addEventListener('touchend', clickInfoBtn);
            window.addEventListener('touchend', clickWindow);
        }
        else {
            hintBtn.addEventListener('click', clickHintBtn);
        }
    }
}