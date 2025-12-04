import { showTooltip } from '../utils/document_utils.js';


export class Hint {
    constructor() {
        this.el = document.querySelector('#hint-btn');

        this.tree = {};
        this.guess = null;
    }


    init(variables) {
        this.getVariables(variables);
        this.createEvents();
    }


    getVariables(variables) {
        this.cfg = variables.cfg;

        this.tree = variables.dataset.guessTree;
    }


    createEvents() {
        const eventType = this.cfg.touch ? 'touchend' : 'click';

        // Click info button event
        const clickHintBtn = () => {
            showTooltip(this.el, `Guess: ${this.guess}`);
        };

        // Create events
        this.el.addEventListener(eventType, clickHintBtn);
    }


    solve(feedback=null) {
        let currentNode = null;
        let successors = null;
        let nextNode = null;

        // Solve get tree node depending on feedback
        if (!feedback) nextNode = this.tree.root;
        else {
            currentNode = this.tree.currentNode;
            successors = this.tree.nodes[currentNode].successors;
            nextNode = successors[feedback];
        }

        // Get word of node
        this.guess = this.tree.nodes[nextNode].word;

        this.tree.currentNode = nextNode;
    }
}