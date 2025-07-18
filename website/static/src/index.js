import { Game } from './classes/game.js';
import { Dataset } from './classes/dataset.js';


const dataset = new Dataset();
const game = new Game();

const variables = { dataset, game }


await dataset.init();
game.init(variables);