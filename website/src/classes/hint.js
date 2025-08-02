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
        this.message = variables.message;

        this.tree = variables.dataset.guessTree;
    }


    createEvents() {
        // Click info button event
        const clickHintBtn = () => {
            this.message.show(`Word hint: <span class="msg-highlight">${this.guess}</span>`);
        };

        // Create events
        if (this.cfg.touch) {
            this.el.addEventListener('touchend', clickHintBtn);
        }
        else {
            this.el.addEventListener('click', clickHintBtn);
        }
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
            console.log(successors, feedback)
            nextNode = successors[feedback];
        }

        // Get word of node
        this.guess = this.tree.nodes[nextNode].word;

        this.tree.currentNode = nextNode;
    }
}