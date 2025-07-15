import { Game } from './classes/game.js';
import { Python } from './classes/python.js';


// List of possible words (you can expand this significantly)
const ALL_WORDS = [
    "apple", "baker", "crane", "dream", "earth", "flame", "grape", "house", "igloo", "jolly",
    "kneel", "lemon", "mango", "night", "ocean", "plant", "queen", "river", "sugar", "table",
    "unity", "virus", "waste", "xerox", "youth", "zebra", "abode", "brave", "charm", "drain",
    "excel", "fresh", "giant", "happy", "ideal", "jumps", "knits", "light", "magic", "novel",
    "outer", "peace", "query", "route", "speak", "train", "ultra", "vocal", "watch", "xylos",
    "yield", "zoned", "amber", "blush", "crest", "dwarf", "elite", "fable", "gleam", "hasty",
    "ivory", "jokes", "kiosk", "lunar", "mirth", "niche", "oasis", "prism", "quart", "relay"
];

const game = new Game(ALL_WORDS);
const python = new Python();

python.init();
window._runPythonFile = async (filepath) => await python.run(filepath);
game.start();