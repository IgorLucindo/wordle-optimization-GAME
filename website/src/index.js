import { Dataset } from './classes/dataset.js';
import { Board } from './classes/board.js';
import { Keyboard } from './classes/keyboard.js';
import { Message } from './classes/message.js';
import { Hint } from './classes/hint.js';
import { Game } from './classes/game.js';
import { Header } from './classes/header.js';


const cfg = {
  touch: 'ontouchstart' in window || navigator.maxTouchPoints > 0,
}
const dataset = new Dataset();
const board = new Board();
const keyboard = new Keyboard();
const message = new Message();
const hint = new Hint();
const game = new Game();
const header = new Header();

const variables = { cfg, dataset, board, keyboard, message, game, header, hint }


await dataset.init();
board.init(variables);
keyboard.init(variables);
hint.init(variables);
game.init(variables);
header.init(variables);