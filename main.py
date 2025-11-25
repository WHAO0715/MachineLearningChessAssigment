#model free --reinforcement learning -- mini alphaZero-style algorithms

import math
import random
import collections
from copy import deepcopy
from typing import Dict, List, Tuple
import os

import chess
import numpy as np #fast mathematical operations on arrays.
#for building and training neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange


#Flow explained step by step:
#ChessEnv:  Provides the current board state, encodes it into a 3D array suitable for the neural network.
#Net:       Takes encoded board → outputs policy probabilities for each move + value estimate for the board.
#MCTS:      Uses NN outputs to simulate possible futures → computes improved policy vector based on visit counts.
#Self-play: Chooses moves using MCTS → stores (obs, pi, z) in replay buffer.
#Training:  Samples from replay buffer → trains Net to better predict policy and value.

class ChessEnv:
    def __init__(self):
        #chess.Board is class from python chess
        #create a object self.board to store information
        self.board = chess.Board()
        self.action_size = 64 * 64 * 7 # 64 grid x 64 grid (all possible movement from one state to other state,5 is one normal move type + pawn(promotion to rook/knight/bishop/queen) )
                                       # each number will represented a movement of chess from one grid to another grid
        self.observation_shape = (8, 8, 12) #8x8 the size chess board x12 is 12 ,different chess (6 type white chess and 6 type black chess ) basic will be the square but in chess the 8x8 represented the position and 12 represented the chess type(for neural net input)
                                            #use tupple(always fixed) instead of list(can change)
    def reset(self):
        self.board.reset() #reset the board
        return self.encode_board() #encode the board for neural net ,return (8,8,12)

    #one move in game
    def step(self, action_idx: int):
        move = self.index_to_move(action_idx)
        if move not in self.board.legal_moves:
            # illegal move: large negative reward -> terminate
            reward = -1.0
            done = True
            return self.encode_board(), reward, done, {}
        #else legal
        self.board.push(move)
        #check current chess alrd end or not, will return true if game end
        done = self.board.is_game_over()
        if done:
            if self.board.outcome().winner is None:
                reward = 0.0  # draw
            else:
                #reward=1.0 when white win and -1.0 when black win
                reward = 1.0 if self.board.outcome().winner == chess.WHITE else -1.0
        return self.encode_board(), reward, done, {}


    def encode_board(self) -> np.ndarray: #just notice return the n-dimensional array same as # returns a NumPy array ,just more formal
        #create 12type chess(6white type and black type chess) of empty array of chess board (represented 1 as presence and 0 as absence)
        #so will have 12 of 8x8 chess board, later will combine
        planes = np.zeros(self.observation_shape, dtype=np.int8) #8-bit integer small memory gud for storing 0,1
        #piece_map return the all sqaure(grid) that have piece,hence at start will loop 32 time -- 32 chess piece
        #item()get the same of both sq and piece together
        for sq, piece in self.board.piece_map().items():
            #        column0  column1 column2.........
            # row7       0       1        2       3  4  5  6  7
            # row6       8       9       10      11 12 13 14 15
            # .....

            #when sq = 0(for loop until 63)
            #row 7-(0//8) = row 7
            #colomn 0%7 = coloumn 0
            row = 7 - (sq // 8)
            col = sq % 8
    #
            # piece.piece_type -- integer value of 1 to 6 that represented the type of chess , -1 cuz start from 0
            #PieceType: TypeAlias = int
            #PAWN: PieceType = 1
            #KNIGHT: PieceType = 2
            #BISHOP: PieceType = 3
            #ROOK: PieceType = 4
            #QUEEN: PieceType = 5
            #KING: PieceType = 6
            plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
            planes[row, col, plane] = 1
        return planes

    #convert real move to index
    #chess move is any legal move such as a1a2 means move from a1 to a2
    def move_to_index(self, move: chess.Move) -> int:  # return the int value
        # from a1a2 change to number (0-63) like a1=0 a2=8 by function chess
        from_sq = move.from_square
        to_sq = move.to_square
        # promotion value 2  (knight),3  (bishop),4  (rook),5  (queen)
        promo = move.promotion if move.promotion is not None else 0
        return from_sq * (64 * 5) + to_sq * 5 + promo

    # convert index(numebr) to real move
    def index_to_move(self, index: int) -> chess.Move:
        from_sq = index // (64 * 5)
        rem = index % (64 * 5)
        to_sq = rem // 5
        promo = rem % 5
        promo = promo if promo != 0 else None
        return chess.Move(from_sq, to_sq, promotion=promo)

    def legal_action_mask(self) -> np.ndarray:
        #create all action array--total 20480
        mask = np.zeros(self.action_size, dtype=np.float32)
        #loop for all legal move,legal_move return all legal move
        for m in self.board.legal_moves:
            idx = self.move_to_index(m)
            #double check the idx is between the range
            if 0 <= idx < self.action_size:
                #1 means legal
                mask[idx] = 1.0
        return mask

#----------------------------------------------------
# Neural Network (predict Policy and Value--1 is )
class Net(nn.Module): #here is inheritance ,so class net can get all method of nn.module
    def __init__(self, action_size):
        super().__init__()
        #12 board (8x8) ,each board represented a type of chess
        in_ch = 12
        #3 convolutional layer is enough for extract the pattern for 8x8 chess
        self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1) #padding=1 keep the output as same input 8x8
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)  #128 is enough too many or too little channel will make it cannot study anything or training time too long
                       #Conv2d(in_channel, out_channel, kernelsize (3 = 3x3 window size) -- focus on local pattern)

        #take the Flattened feature map from last conv layer
        # 128 is 128 channel from conv layer, 8x8 is the chess board size, then compress all 8192 feature(128x128x8) into 512 number that use for policy and value
        self.fc_shared = nn.Linear(128 * 8 * 8, 512)

        # policy head
        self.policy_head = nn.Linear(512, action_size)

        # value head
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        #Converts input from a NumPy array to a PyTorch tensor
        # x: (B, 8,8,12) numpy -> needs to be (B,12,8,8)
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.permute(0, 3, 1, 2)  # (B,12,8,8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) #become flattening
        x = F.relu(self.fc_shared(x))

        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x)).squeeze(-1)
        #return current state of best choose and expected result value
        return policy_logits, value

# each mcts node represented a state of board and it possible action
class MCTSNode:
    def __init__(self, prior: float = 0.0):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0  # total value from this node
        self.children: Dict[int, "MCTSNode"] = {}  # action -> node

    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class MCTS:
    def __init__(self, net: Net, env: ChessEnv, device='cpu', c_puct=1.0, n_simulations=25):
        self.net = net
        self.env = env
        self.device = device
        #can adjust--
        self.c_puct = c_puct #Controls the balance between exploration and exploitation when selecting which child node to visit in MCTS.
                             #high--explore less-visited moves more aggressively / low--follow the nn sugested policy
        self.n_simulations = n_simulations #how many time simulate the selection action from that state

    def run(self, board: chess.Board):
        """
        Run MCTS from current board position and return policy vector (visit counts normalized)
        """
        root = MCTSNode()
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 0:
            return np.zeros(self.env.action_size, dtype=np.float32)

        # Use network to initialize priors
        obs = self._board_to_obs(board)
        with torch.no_grad(): # torch.no_grad() means no training , just pass forward the network
            #logits-- policy (scores for all possible actions)
            logits, _ = self.net(obs.to(self.device)) #use cpu or gpu
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0] #convert the score logits to probabilities between 0 and 1,and also change the format torch back to numpy array
                                                               #if not mistake will got 20480 number prob in the array (64x64 x5(5 promotion))--each action means one movement

        # initialize all children from legal moves
        for m in legal_moves:
            idx = self.env.move_to_index(m)
            prior = float(probs[idx])
            root.children[idx] = MCTSNode(prior=prior)

        # run simulations
        for _ in range(self.n_simulations): #if 1 means run 1 time only for this state, so more simulation can find better policy value
            self._simulate(root, board.copy())

        # build policy from visit counts
        visit_counts = np.zeros(self.env.action_size, dtype=np.float32)
        for a, node in root.children.items():
            visit_counts[a] = node.visit_count
        if visit_counts.sum() == 0:
            return visit_counts
        return visit_counts / visit_counts.sum()

    def _simulate(self, root: MCTSNode, board: chess.Board):
        #a simulation -- selection (choose best one),expansion and evaluation, then backup, loop until game end
        path = []
        node = root
        cur_board = board.copy()
        # selection
        while True:
            if len(node.children) == 0:
                break
            # choose action maximizing U + Q (balance between exploration and explotation)--one is know haven't try so many time this path, maybe better, one is know from the past this path is good
            total_visit = sum(child.visit_count for child in node.children.values()) #visit_count means how many time the child node is visted for simulation=total time of exploration for this node
            best_score = -float('inf')
            best_action = None
            best_child = None
            for a, child in node.children.items():
                q = child.q_value()
                u = self.c_puct * child.prior * math.sqrt(total_visit + 1e-8) / (1 + child.visit_count)
                score = q + u
                if score > best_score:
                    best_score = score
                    best_action = a
                    best_child = child
            # apply best_action to board
            move = self.env.index_to_move(best_action)
            if move not in cur_board.legal_moves:
                # illegal due to board differences — treat as zero-value
                value = 0.0
                self._backup(path + [(best_action, best_child)], value)
                return
            cur_board.push(move) #when make a move will auto change the site(e.g. white to black)
            path.append((best_action, best_child))
            node = best_child
            # check the game end already or not (terminal)
            if cur_board.is_game_over():
                winner = cur_board.outcome().winner
                if winner is None:
                    value = 0.0
                else:
                    value = 1.0 if winner == chess.WHITE else -1.0
                self._backup(path, value)
                return

        # expansion + evaluation
        # evaluate leaf with network
        obs = self._board_to_obs(cur_board)
        with torch.no_grad():
            logits, value = self.net(obs.to(self.device))
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            value = float(value.cpu().numpy()[0])

        # add children for legal moves
        for m in cur_board.legal_moves:
            idx = self.env.move_to_index(m)
            if idx not in node.children:
                node.children[idx] = MCTSNode(prior=float(probs[idx]))
        # backup
        self._backup(path, value)

    def _backup(self, path: List[Tuple[int, MCTSNode]], value: float):
        # propagate value up path (from leaf to root)
        for action, node in reversed(path):
            node.visit_count += 1
            # value here is from current player's perspective, flip sign when going up
            node.total_value += value
            value = -value

    def _board_to_obs(self, board: chess.Board) -> torch.Tensor:
        env_copy = deepcopy(self.env) #make a copy so no change the ori one
        env_copy.board = board #make the current state of board to env_copy.board
        obs = env_copy.encode_board()
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # unsqueeze(0) --add a dimension from 3d to 4d e.g. (1,8,8,12) here 1 represented the first board
                                                                   # purpose for nn only can in torch format instead of array format (encode_board)
        return obs




# ---------------------
# Self-play and training utilities
# ---------------------
#obs is encoded board state of 8x8(board chess size) x12(type chess)
GameExample = collections.namedtuple('GameExample', ['obs', 'pi', 'z'])

#store memory of self-play game
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity #max number of game state to store
        self.buffer: List[GameExample] = [] #to store the all game state in list

    def add_game(self, game_examples: List[GameExample]):
        self.buffer.extend(game_examples)
        if len(self.buffer) > self.capacity:
            # drop oldest ,take only newest 10000
            self.buffer = self.buffer[-self.capacity:]


    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        #np.stack is make the multiple array to one array
        obs_list = np.array([x.obs for x in batch], dtype=np.float32) #observation--here make the multiple(depend on how many batch size)3d array of 8x8x12 to one array that combining all to one 4d array
                                                                      #(e.g. (20,8,8,9)=1 means in number 20 of state that selected of (8,8) the type 9 maybe is black king , =1 means presence
        pi_list = np.array([x.pi for x in batch], dtype=np.float32) #policy
        #obs_list = np.stack([ex.obs for ex in batch], axis=0)
        #pi_list = np.stack([ex.pi for ex in batch], axis=0)
        z_list = np.array([x.z for x in batch], dtype=np.float32) #value--game expected outcome (+1 current player win ,0 draw ,-1 current player lose)
        return obs_list, pi_list, z_list


def self_play_game(env: ChessEnv, net: Net, mcts: MCTS, temperature=1.0, max_moves=500):
    board = chess.Board()
    examples = []
    while not board.is_game_over() and board.fullmove_number < max_moves:
        # run MCTS
        policy = mcts.run(board)  # vector of probs
        if policy.sum() == 0:
            # fallback to legal moves uniform
            mask = np.zeros(env.action_size, dtype=np.float32)
            for m in board.legal_moves:
                mask[env.move_to_index(m)] = 1.0
            policy = mask / mask.sum()

        # apply temperature
        if temperature == 0:
            # choose argmax
            action_idx = int(np.argmax(policy))
        else:
            # sample proportionally
            probs = policy ** (1.0 / temperature)
            probs = probs / probs.sum()
            action_idx = int(np.random.choice(len(probs), p=probs))

        # record example: obs and target pi (the visit distribution)
        obs = encode_board_from_board(board, env)
        examples.append(GameExample(obs=obs, pi=policy.copy(), z=None))

        # make move
        move = env.index_to_move(action_idx)
        if move not in board.legal_moves:
            # illegal — immediate loss for side to move
            # set z later appropriately
            board.result()  # do nothing
            break
        board.push(move)

    # determine game outcome in perspective of each example
    outcome = board.outcome()
    if outcome is None:
        final_z = 0.0
    else:
        winner = outcome.winner
        if winner is None:
            final_z = 0.0
        else:
            final_z = 1.0 if winner == chess.WHITE else -1.0

    # assign z to each example from viewpoint of the player who was to move at that state
    examples_with_z = []
    b = chess.Board()
    for ex in examples:
        # the board before the move corresponds to the side to move stored in ex.obs
        # we use ply parity to know side: ex.obs doesn't include turn explicitly; we can reconstruct by
        # applying moves sequentially — simpler: keep a board sequence when creating examples.
        # For simplicity, we will alternate signs starting from white:
        # assume the first example is white to move, then next is black, ...
        examples_with_z.append(ex)
    # assign z alternating sign: first example z = final_z, next = -final_z, etc.
    z_list = []
    for i in range(len(examples_with_z)):
        sign = 1 if (i % 2 == 0) else -1
        z_list.append(final_z * sign)
    result_examples = []
    for ex, z in zip(examples_with_z, z_list):
        result_examples.append(GameExample(obs=ex.obs, pi=ex.pi, z=z))
    return result_examples, outcome

def encode_board_from_board(board: chess.Board, env: ChessEnv):
    env_copy = deepcopy(env)
    env_copy.board = board
    return env_copy.encode_board()


# lr=learning rate (lower more stable learning but more long time) (can change the value for purpose tuning)
def train(network: Net, replay: ReplayBuffer, epochs=1, batch_size=64, lr=1e-3, device='cpu'):
    if len(replay.buffer) == 0:
        return
    #use adam optimizer,give all things inside nn (include conv layer,flatenning,policy,value..) and learning rate for adam optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    #function come from library pytorch
    network.to(device) #can select use cpu or gpu
    network.train()

    for _ in range(epochs):
        #here of batch size is means take the 64(can change) random state maybe some is the state of step 13 of first game , other one might be the state of 30step of third game
        obs, pi, z = replay.sample(batch_size)
        obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
        pi_t = torch.tensor(pi, dtype=torch.float32).to(device)
        z_t = torch.tensor(z, dtype=torch.float32).to(device)

        logits, value = network(obs_t)
        value_loss = F.mse_loss(value, z_t)
        # policy loss: cross-entropy between target pi and logits
        log_probs = F.log_softmax(logits, dim=-1)
        policy_loss = - (pi_t * log_probs).sum(dim=1).mean()

        loss = value_loss + policy_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate_vs_random(env: ChessEnv, net: Net, mcts_sim=25, games=10, device='cpu'):
    mcts = MCTS(net, env, device=device, n_simulations=mcts_sim)
    wins = 0
    draws = 0
    losses = 0
    for g in range(games):
        board = chess.Board()
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                # agent that already train (white)
                policy = mcts.run(board)
                if policy.sum() == 0:
                    # fallback
                    legal = [env.move_to_index(m) for m in board.legal_moves]
                    a = random.choice(legal)
                else:
                    a = int(np.argmax(policy))
                move = env.index_to_move(a)
                if move not in board.legal_moves:
                    # illegal — lose
                    break
                board.push(move)
            else:
                # random black move
                move = random.choice(list(board.legal_moves))
                board.push(move)
            if board.fullmove_number > 300:
                break
        outcome = board.outcome()
        if outcome is None:
            draws += 1
        else:
            if outcome.winner is None:
                draws += 1
            elif outcome.winner == chess.WHITE:
                wins += 1
            else:
                losses += 1
    return wins, draws, losses


# Main --- training loop
def main2():

    if os.path.exists("mini_alphazero_net.pth"):
        print("Pretrained model found. Skipping training.")
        return
    
    device = 'cpu'
    env = ChessEnv()
    net = Net(action_size=env.action_size)
    replay = ReplayBuffer()

    mcts = MCTS(net, env, device=device, c_puct=1.0, n_simulations=5)

    n_iterations = 5  #how many times repeat the full training loop
    games_per_iter = 1  # how many self-play games per iteration
    train_epochs = 1
    print("Starting Mini-AlphaZero training ")
    for it in trange(n_iterations):
        # self-play
        for g in range(games_per_iter):
            examples, outcome = self_play_game(env, net, mcts, temperature=1.0)
            replay.add_game(examples)
        # train
        train(net, replay, epochs=train_epochs, batch_size=64, lr=1e-3, device=device)

        # periodic eval
        if it % 5 == 0:
            w, d, l = evaluate_vs_random(env, net, mcts_sim=10, games=10, device=device)
            print(f"Iter {it:03d} eval vs random -> wins:{w} draws:{d} losses:{l}")

        print(f"\n=== Self-play Game {it+1} ===")

    # save model
    torch.save(net.state_dict(), "mini_alphazero_net.pth")
    print("Training complete. Model saved to mini_alphazero_net.pth")

