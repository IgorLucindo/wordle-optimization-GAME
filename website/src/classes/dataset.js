export class Dataset {
    constructor() {
        this.words = [];
        this.solutionWords = [];
    }


    async init() {
        await this.getWords();
    }


    async getWords() {
        const [res1, res2] = await Promise.all([
            fetch('../datasets/words/solutions.txt'),
            fetch('../datasets/words/non_solutions.txt')
        ]);
        const [data1, data2] = await Promise.all([res1.text(), res2.text()]);

        const words1 = data1.split(/\r?\n/).filter(line => line.trim() !== '');
        const words2 = data2.split(/\r?\n/).filter(line => line.trim() !== '');

        this.solutionWords = words1;
        this.words = words1.concat(words2);
    }
}