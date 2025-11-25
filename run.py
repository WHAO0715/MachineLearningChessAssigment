import pygame as p
import chess
import time
import random
import torch
import os
from main import ChessEnv, Net, MCTS

WIDTH = HEIGHT = 512
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES = {}

def loadImages():

    pieces = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']
    for piece in pieces:
        path = os.path.join("images", piece + ".png")
        if os.path.exists(path):
            IMAGES[piece] = p.transform.scale(p.image.load(path), (SQ_SIZE, SQ_SIZE))
        else:
            print(f"Warning: Image not found for {piece} at {path}")

def drawGameState(screen, board):

    drawBoard(screen)
    drawPieces(screen, board)

def drawBoard(screen):

    colors = [p.Color("white"), p.Color("gray")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[((r + c) % 2)]
            p.draw.rect(screen, color, p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def drawPieces(screen, board):

    for square_index in range(64):
        piece = board.piece_at(square_index)
        if piece:

            symbol = piece.symbol() # 'P', 'p', 'k', 'K'
            color_prefix = 'w' if piece.color == chess.WHITE else 'b'
            piece_key = color_prefix + symbol.upper()
            

            file_idx = chess.square_file(square_index)
            rank_idx = chess.square_rank(square_index)
            
            # Draw
            if piece_key in IMAGES:
                screen.blit(IMAGES[piece_key], p.Rect(file_idx * SQ_SIZE, (7 - rank_idx) * SQ_SIZE, SQ_SIZE, SQ_SIZE))


def main():
    p.init()
    screen = p.display.set_mode((WIDTH, HEIGHT))
    p.display.set_caption("AlphaZero vs Human")
    clock = p.time.Clock()
    screen.fill(p.Color("white"))
    
    loadImages()
    
    device = 'cpu' # Use 'cuda' if available
    env = ChessEnv()
    net = Net(action_size=env.action_size)

    if os.path.exists("mini_alphazero_net.pth"):
        print("Loading pretrained AlphaZero model...")
        net.load_state_dict(torch.load("mini_alphazero_net.pth", map_location=device))
        net.eval()
    else:
        print("Model file not found! Starting with random weights.")

    mcts = MCTS(net, env, device=device, n_simulations=25)

    board = chess.Board()
    running = True
    game_over = False
    
    sq_selected = () # (row, col)
    player_clicks = [] # keep track of player clicks [(row, col), (row, col)]

    print("Game Started. You play as Black (Standard Pygame controls).")

    while running:
        human_turn = (board.turn == chess.BLACK)
        
        # 1. Event Handling (Mouse clicks and Quit)
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False
            
            elif e.type == p.MOUSEBUTTONDOWN:
                if not game_over and human_turn:
                    location = p.mouse.get_pos() # (x, y)
                    col = location[0] // SQ_SIZE
                    row = location[1] // SQ_SIZE
                    
                    if sq_selected == (row, col): 
                        sq_selected = ()
                        player_clicks = []
                    else:
                        sq_selected = (row, col)
                        player_clicks.append(sq_selected)
                    
                    if len(player_clicks) == 2:
  
                        start_row, start_col = player_clicks[0]
                        end_row, end_col = player_clicks[1]
                        
                        start_sq = chess.square(start_col, 7 - start_row)
                        end_sq = chess.square(end_col, 7 - end_row)
                        
                        move = chess.Move(start_sq, end_sq)
                        
    
                        piece = board.piece_at(start_sq)
                        if piece and piece.piece_type == chess.PAWN:
                            if (board.turn == chess.WHITE and 7 - end_row == 7) or \
                               (board.turn == chess.BLACK and 7 - end_row == 0):
                                move = chess.Move(start_sq, end_sq, promotion=chess.QUEEN)

                        if move in board.legal_moves:
                            board.push(move)
                            print(f"Human plays: {move.uci()}")
                            sq_selected = ()
                            player_clicks = []
                        else:
                            print("Illegal move, try again.")
                            player_clicks = [sq_selected]

        # 2. AI Turn Logic
        if not game_over and not human_turn:

            drawGameState(screen, board)
            p.display.flip()
            
            print("AI is thinking...")
            policy = mcts.run(board)
            
            if policy.sum() == 0:
                legal = [env.move_to_index(m) for m in board.legal_moves]
                a = random.choice(legal)
            else:
                a = int(policy.argmax())
                
            move = env.index_to_move(a)
            print(f"AI plays: {move.uci()}")
            board.push(move)

        # 3. Check Game Over
        if board.is_game_over():
            game_over = True
            outcome = board.outcome()
            # Draw one last time to show final state
            drawGameState(screen, board) 
            p.display.flip()
            
            if outcome.winner is None:
                print("Game Over: Draw")
            elif outcome.winner == chess.WHITE:
                print("Game Over: AI Wins")
            else:
                print("Game Over: You Win")
            
            time.sleep(5)
            running = False

        # 4. Drawing
        drawGameState(screen, board)
        clock.tick(MAX_FPS)
        p.display.flip()

    p.quit()

if __name__ == "__main__":
    main()