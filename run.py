import chess
import chess.svg
from IPython.display import SVG, display, clear_output
import time
import random
import torch
from main import ChessEnv 

def show_board(board: chess.Board):
    """
    Display the chess board in Colab using SVG
    """
    clear_output(wait=True)
    display(SVG(chess.svg.board(board=board, size=400)))

def play_vs_human_with_images():

    device = 'cpu'
    env = ChessEnv()
    net = Net(action_size=env.action_size)
    net.load_state_dict(torch.load("mini_alphazero_net.pth", map_location=device))
    net.eval()
    mcts = MCTS(net, env, device=device, n_simulations=25)

    board = chess.Board()
    print("Welcome! You play as Black. Enter moves in UCI format (e2e4, g1f3, etc.)")
    show_board(board)

    while not board.is_game_over():
        # AI move
        if board.turn == chess.WHITE:
            policy = mcts.run(board)
            if policy.sum() == 0:
                legal = [env.move_to_index(m) for m in board.legal_moves]
                a = random.choice(legal)
            else:
                a = int(policy.argmax())
            move = env.index_to_move(a)
            print(f"AI plays: {move.uci()}")
            board.push(move)
            show_board(board)
            time.sleep(1)

        # Human move
        else:
            print(f"Legal moves: {[move.uci() for move in board.legal_moves]}")

            move_uci = input("Your move: ").strip()

            try:
                move = chess.Move.from_uci(move_uci)

                if move not in board.legal_moves:
                    print("Illegal move! Try again.")
                    continue
                board.push(move)
                show_board(board)
                time.sleep(1)
            except:
                print("Invalid input! Use UCI format, e.g., e2e4")
                continue

    print("Game over!")
    outcome = board.outcome()
    if outcome.winner is None:
        print("Draw!")
    elif outcome.winner == chess.WHITE:
        print("AI wins!")
    else:
        print("You win!")

play_vs_human_with_images()
