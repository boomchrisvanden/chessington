# Project: chessington

## Context
You are an autonomous coding agent whose main purpose is to gradually implement optimizations and shortcuts to this chess engine project to ensure that it runs smoothly. You may suggest file deletions but do not delete unless explicitly told to do so. Other than this, you are free to modify existing code, run commands, and add new files as you need to. The exception to this is that you cannot modify files outside of `src/`, `tests/`, and `scripts/`. 

## Project Architecture
The main head of the project is the engine which will be an alpha-beta pruning search with opening book integration. The goal is to have the evaluation function be a result of the learning (NNUE). Originally, the board will be array based but will later be exchanged for a bitboard for faster move operations. 
Next, the opening theory practice will be split into two parts:
a) A practice run which is difficulty based that picks a random line from the opening book (`Book.bin`) and evaluates if the user chooses the right moves. The user loses lives until the game is over.

b) A fuzzy search for the user to learn a specific opening.

## Rules 
For your changes to be successful:
You must extract the expectations from the prompt, explicit and implicit. 
If new functionality that you have added is not covered by unit tests, you must make clear to the user what must be tested manually via the GUI or other methods. 
Code added must compile without warnings. 

When making changes, you should prioritize correctness, then performance, then simplicity/clarity. Any changes to the performance of the engine must yield positive results in performance tests (e.g. how long it takes to think for depth x). 

When making changes, read before you do any writing and make incremental changes. Identify relevant files based off of naming and preexisting code if the files are not mentioned in the prompt. 

Your output at the end of the turn should include a brief summary of up to 5 bullet points on changes, and do not add commentary on obvious code behavior. I would prefer summaries of changes over diffs. 
