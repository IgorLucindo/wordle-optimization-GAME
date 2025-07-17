import { Game } from './classes/game.js';


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

// --- Streamlit Component Integration START ---
// Streamlit.setComponentReady();
// Streamlit.setFrameHeight();
// --- Streamlit Component Integration END ---

const game = new Game(ALL_WORDS);

game.start();