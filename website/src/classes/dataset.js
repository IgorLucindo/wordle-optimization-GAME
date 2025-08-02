export class Dataset {
    constructor() {
        this.allWords = [];
        this.keyWords = [];
        this.guessTree = {};
    }


    async init() {
        await this.getWords();
        await this.getGuessTree();
    }


    async getWords() {
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


    async getGuessTree() {
        const response = await fetch('./dataset/guess_tree.json');
        this.guessTree = await response.json();
    }
}