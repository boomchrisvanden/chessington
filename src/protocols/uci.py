'''
Author: Chris Vanden Boom
11/18/25
'''
import sys

def uci_loop():
    
    '''
    Parse line into tokens
    Detect:
        uci
            - id name 
        isready
            - reply readok (after ready)

        ucinewgame
            - new game is starting
        position
            - position startpos
            - position fen
            - position startpos moves e2e4 e7e5 ... 
        go
            - go depth N
            - when search is done, output bestmove <best move> 
        stop
            - interrupt search, print a bestmove with best found so far. 
        quit 
            - exit loop and terminate
        setoption 

    '''

    # todo: setup stuff
    Board = None
    tt = TranspositionTable(size_mb=32)

    options = {} # for setoption command
    printf("Entering UCI loop. Type 'help' for a list of commands")

    # Enter loop
    for line in sys.stdin:
        line = line.strip()

        if not line:
            continue

        tokens = line.split()
        cmd = tokens[0]

        if cmd == "uci":
            print("id name chessington")
            print("id author Chris Vanden Boom")
            print("uciok")
            sys.stdout.flush()

        elif cmd == "isready":

            print("readyok")
            sys.stdout.flush()

        elif cmd == "ucinewgame":
            # clear tt, stats, etc.
            tt = TranspositionTable(size_mb=32)

        elif cmd == "position":
            # TODO: implement
            #board = handle_position(tokens[1:])
            pass

        elif cmd == "setoption":
            handle_setoption(tokens[1:], options)

        elif cmd == "go":
            if board is None:
                continue

            limits = parse_go(tokens[1:0])
            max_depth = limits.depth or 5
            time_ms = limits.movetime or 0
            score, best_move, depth = iterative_deepening(board, max_depth, time_ms, tt)
            print(f"bestmove {best_move.uci()}")
            sys.stdout.flush()

        elif cmd == "stop":
            pass # unsupported so far

        elif cmd == "quit":
            break

        else:
            #ignore
            pass

    return

def print_help():
    printf(
        '''
        uci
            - id name 
        isready
            - reply readok (after ready)

        ucinewgame
            - new game is starting
        position
            - position startpos
            - position fen
            - position startpos moves e2e4 e7e5 ... 
        go
            - go depth N
            - when search is done, output bestmove <best move> 
        stop
            - interrupt search, print a bestmove with best found so far. 
        quit 
            - exit loop and terminate
        setoption 
        '''
    )
    