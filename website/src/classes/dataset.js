export class Dataset {
    constructor() {
        this.allWords = [];
        this.keyWords = [];
        this.tree = {};
        this.tree_hard = {};
    }


    async init() {
        await this.loadWords();
        await this.loadTree();
    }


    async loadWords() {
        const [res1, res2] = await Promise.all([
            fetch('./dataset/solutions.txt'),
            fetch('./dataset/non_solutions.txt')
        ]);
        const [data1, data2] = await Promise.all([res1.text(), res2.text()]);

        const words1 = data1.split(/\r?\n/).filter(line => line.trim() !== '');
        const words2 = data2.split(/\r?\n/).filter(line => line.trim() !== '');

        this.keyWords = words1;
        this.allWords = words1.concat(words2);
    }


    async loadTree() {
        const tree_response = await fetch('./dataset/decision_tree.json');
        const tree_hard_response = await fetch('./dataset/decision_tree_hard.json');

        this.tree = await tree_response.json();
        this.tree_hard = await tree_hard_response.json();
    }
}