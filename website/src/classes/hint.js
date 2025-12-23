import { showTooltip } from '../utils/document_utils.js';


export class Hint {
    constructor() {
        this.el = document.querySelector('#hint-btn');

        this.tree = {};
        this.tree_hard = {};
        this.currentNode = null;
        this.guess = null;
    }


    init(variables) {
        this.getVariables(variables);
        this.createEvents();
        showTooltip(this.el, "Click here to see our calculated best guess.", 6);
    }


    getVariables(variables) {
        this.cfg = variables.cfg;
        this.game = variables.game;

        this.tree = variables.dataset.tree;
        this.tree_hard = variables.dataset.tree_hard;
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
        const activeTree = this.game.hardmode ? this.tree_hard : this.tree;

        // Update current node
        if (!feedback) {
            this.currentNode = activeTree.root;
        } else {
            const feedbackTuple = `(${feedback.join(', ')})`;
            const key = `(${this.currentNode}, ${feedbackTuple})`;
            this.currentNode = activeTree.successors[key];
        }

        // Get guess of node
        this.guess = activeTree.nodes[this.currentNode].guess.toUpperCase();
    }
}