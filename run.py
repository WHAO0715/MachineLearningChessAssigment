import pygame as p
import chess
import time
import random
import torch
import os
from main import ChessEnv, Net, MCTS

# --- Configuration ---
WIDTH = HEIGHT = 512
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES = {}

# --- Pygame Helper Functions ---

def loadImages():
    """Loads images into the global dictionary."""
    pieces = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']
    for piece in pieces:
        path = os.path.join("images", piece + ".png")
        if os.path.exists(path):
            IMAGES[piece] = p.transform.scale(p.image.load(path), (SQ_SIZE, SQ_SIZE))
        else:
            print(f"Warning: Image not found for {piece} at {path}")

def drawGameState(screen, board):
    """Responsible for all graphics within a current game state."""
    drawBoard(screen)
    drawPieces(screen, board)

def drawBoard(screen):
    """Draw the squares on the board."""
    colors = [p.Color("white"), p.Color("gray")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[((r + c) % 2)]
            p.draw.rect(screen, color, p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def drawPieces(screen, board):
    """Draw the pieces on the board using python-chess board state."""
    # python-chess squares are 0-63. A1 is 0, H1 is 7, A8 is 56, H8 is 63.
    # Pygame renders (0,0) at top-left.
    
    for square_index in range(64):
        piece = board.piece_at(square_index)
        if piece:
            # Convert piece to string key, e.g., 'wP', 'bK'
            symbol = piece.symbol() # 'P', 'p', 'k', 'K'
            color_prefix = 'w' if piece.color == chess.WHITE else 'b'
            piece_key = color_prefix + symbol.upper()
            
            # Calculate coordinates
            # File (col) is square_index % 8
            # Rank (row) is square_index // 8
            # But we need to flip the rank because Pygame y=0 is top
            file_idx = chess.square_file(square_index)
            rank_idx = chess.square_rank(square_index)
            
            # Draw
            if piece_key in IMAGES:
                screen.blit(IMAGES[piece_key], p.Rect(file_idx * SQ_SIZE, (7 - rank_idx) * SQ_SIZE, SQ_SIZE, SQ_SIZE))

# --- Main Game Logic ---

def main():
    p.init()
    screen = p.display.set_mode((WIDTH, HEIGHT))
    p.display.set_caption("AlphaZero vs Human")
    clock = p.time.Clock()
    screen.fill(p.Color("white"))
    
    # Load Images
    loadImages()
    
    # --- AI Setup ---
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

    # --- Game State ---
    board = chess.Board()
    running = True
    game_over = False
    
    # Interaction variables
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
                    
                    if sq_selected == (row, col): # User clicked same square twice
                        sq_selected = ()
                        player_clicks = []
                    else:
                        sq_selected = (row, col)
                        player_clicks.append(sq_selected)
                    
                    # If user selected two squares (Start -> End)
                    if len(player_clicks) == 2:
                        # Convert Pygame coords (row, col) to python-chess Move
                        # Pygame Row 0 = Rank 8. Pygame Row 7 = Rank 1.
                        start_row, start_col = player_clicks[0]
                        end_row, end_col = player_clicks[1]
                        
                        start_sq = chess.square(start_col, 7 - start_row)
                        end_sq = chess.square(end_col, 7 - end_row)
                        
                        move = chess.Move(start_sq, end_sq)
                        
                        # Check for promotion (simple auto-queen for UI simplicity)
                        # If a pawn moves to the last rank, assume promotion to Queen
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
                            # Keep the second click as the new 'first' click allows for easier correction
                            player_clicks = [sq_selected]

        # 2. AI Turn Logic
        if not game_over and not human_turn:
            # We draw the board once before AI thinks so the screen updates the human's last move
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
            
            # Pause to let user see result before closing (optional)
            time.sleep(5)
            running = False

        # 4. Drawing
        drawGameState(screen, board)
        clock.tick(MAX_FPS)
        p.display.flip()

    p.quit()

if __name__ == "__main__":
    main()