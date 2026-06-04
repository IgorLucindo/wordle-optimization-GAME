import { showOverlay, hideOverlay } from '../utils/document_utils.js';


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
        this.game = variables.game;
    }


    createEvents() {
        if (!('ontouchstart' in window) || navigator.maxTouchPoints === 0) return;
        
        const infoBtn = this.el.querySelector('#info-btn');
        const footer = document.querySelector('footer');
        
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
        infoBtn.addEventListener('touchend', clickInfoBtn);
        window.addEventListener('touchend', clickWindow);
    }
}