
import random
import torch
import numpy as np
import math
import time
import logging
import multiprocessing
from typing import Tuple, Optional
from rl_hanabi.game.hle_state import HLEGameState
from rl_hanabi.mcts.convention_rollout import ConventionRolloutPolicy, ParallelConventionRollout
from hanabi_learning_environment.pyhanabi import HanabiMove


logger = logging.getLogger(__name__)
# logging.basicConfig(filename='mcts_log.log', encoding='utf-8', level=logging.DEBUG)



class Node:

    def __init__(self, 
                 state: HLEGameState, 
                 last_action : HanabiMove | None, 
                 parent : 'Node | None',
                 c : float = 0.2,
                 prior : float = 0,
                 mcts: 'MCTS | None' = None) -> None:
        self.state = state
        self.pyhanabi_game = state.game
        self.max_moves = self.state.action_space_size
        self.valid_moves = set(self.state.legal_moves()) # List of HanabiMove
        self.valid_moves_bool = self.state.legal_moves_mask() 
        self.n_visits : int = 0     
        self.expanded_move = set()
        self.parent : Node | None = parent
        self.children_list : list[Node] = []
        self.is_terminal = self.state.is_terminal()
        self.c = c
        self.prior = prior
        self.mcts = mcts
        self.value = 0
        self.last_action = last_action
        self.current_player = self.state.current_player_index
        self.set_prob()

    
    def set_prob(self):
        p = np.zeros(self.max_moves, dtype=float)
        if len(self.valid_moves) > 0:
            # Uniform prior over valid moves
            # Every valid move gets equal probability
            p[self.valid_moves_bool] = 1.0 / len(self.valid_moves)
        self.prob_dist = p

    
    def set_prob_dist(self, prob_dist: np.ndarray):
        prob_dist_norm = prob_dist * self.valid_moves_bool

        if np.sum(prob_dist_norm) > 0:
            prob_dist_norm = prob_dist_norm / np.sum(prob_dist_norm)
        else:
            # Fallback to uniform distribution over valid moves
            prob_dist_norm = np.zeros_like(prob_dist)
            prob_dist_norm[self.valid_moves_bool] = 1.0 / np.sum(self.valid_moves_bool)

        self.prob_dist = prob_dist_norm


    def new_mcts(self):
        if self.mcts is None:
            raise Exception("MCTS instance is None in Node")
        return MCTS(
            model=self.mcts.model,
            device=self.mcts.device,
            time_ms=self.mcts.time_ms,
            root_node=self
        )

    def is_root(self):
        if self.mcts is None:
            raise Exception("MCTS instance is None in Node")
        return self.mcts.root_node == self

    def __repr__(self) -> str:
        state_repr = str(self.state)
        return f"Node(state=\n{state_repr}\n, n_visits={self.n_visits}, value={self.value})"

        
    @property
    def q_value(self):
        if self.n_visits == 0:
            print("Why are you here ??")
            return float('inf')
        return self.value / self.n_visits 
    
    def cpuct(self):
        if self.parent is None:
            raise Exception("Parent is None in cpuct calculation")
        return self.q_value + self.c * self.prior * (math.sqrt(self.parent.n_visits) / (self.n_visits + 1)) 

    @DeprecationWarning
    def children_cpuct(self): 

        if self.n_visits == 0:
            print("Komisch!!!")
            return float('inf')
        # UCT formula positive q value since hanabi is a cooperative game
        return [  child.q_value + self.c  * child.prior * (math.sqrt(self.n_visits) / (child.n_visits + 1)) 
                for child in self.expanded_children]
    
    @property
    def expanded_children(self):
        return self.children_list
    
    @property
    def is_fully_expanded(self):
        return len(self.expanded_children) == len(self.valid_moves)
    
    def expand_node(self):
        if self.is_fully_expanded:
            raise Exception("Trying to Expand fully expanded node or Terminal Node")
            
        else:
            action = random.choice(list(self.valid_moves - self.expanded_move))
            next_state = self.state.copy()
            next_state.apply_move(action)
            prior = self.prob_dist[self.state.move_to_index(action)]

            new_node = Node(state=next_state, last_action=action, parent=self, 
                            c=self.c, prior=prior, mcts=self.mcts)
            self.children_list.append(new_node)
            self.expanded_move.add(action)
            return new_node

            

class MCTS:
    def __init__(self, model,
                 device,
                 time_ms : int = 3000, 
                 root_node : Node | None = None
                 ) -> None:
        self.device = device
        self.model = model
        self.time_ms = time_ms
        self.root_node = root_node 

    # For Modularity root parent and last action of MCTS must not be None
    # Do not use node.parent == None or last_action == None to check for root
    def init_root(self, state: HLEGameState, c : float):
        self.root_node = Node(state, last_action=None, parent=None, c=c, mcts=self)


    # Last move should be sorted from oldest to newest
    def use_subtree(self, last_moves: list[HanabiMove]):
        if self.root_node is None:
            return None

        found = False
        current_node = self.root_node
        while len(last_moves) > 0:
            matched_child = None
            for child in current_node.expanded_children:
                if child.last_action is None:
                    continue
                if HLEGameState.moves_are_equal(child.last_action, last_moves[0]):
                    matched_child = child
                    break
            if matched_child is None:
                found = False
                break
            else:
                if matched_child.is_terminal:
                    raise Exception("Trying to use subtree that is terminal")
                else:
                    current_node = matched_child
                    last_moves.pop(0)
                    if len(last_moves) == 0:
                        found = True
                        break
        
        if found:
            self.root_node = current_node
            return True
        else:
            return False
        

    def select(self):
        if self.root_node is None:
            raise Exception("Root Node is None in MCTS select")
        
        current_node  = self.root_node
        while(True):
            selected_child = None

            for child in current_node.expanded_children:
                if child.is_terminal:
                    return child

            if not current_node.is_fully_expanded:
                return current_node
            
            # Use max with a key function for efficiency
            selected_child = max(current_node.expanded_children, key=lambda child: child.cpuct())
            #selected_child = current_node.expanded_children[np.argmax(current_node.children_cpuct())]
            
            if selected_child is None:
                print("Selected Child is None in select reusing last node")
                return current_node
            
            current_node = selected_child


    def expand(self, node : Node):
        if node.is_fully_expanded or node.is_terminal:
            return node
        
        return node.expand_node()
        
        

    def backprop(self, node : Node, value):

        if node is None:
            raise Exception("Node is None in backprop")
        
        while(not node.is_root()):
            node.n_visits += 1
            node.value += value
            if node.parent is None:
                raise Exception("Parent is None in backprop")
            else:
                node = node.parent

        node.n_visits += 1
        node.value += value

    def decay_stats(self, node, factor=0.5):
        node.n_visits = int(node.n_visits * factor)
        node.value *= factor
        for child in node.children:
            if child is not None:
                self.decay_stats(child, factor)

    def rollout_policy(self, node: Node, rollout_depth: int = 25, 
                       num_rollouts: int = 1,
                       use_parallel: bool = False) -> Tuple[np.ndarray, float]:
        """
        Convention-based rollout policy using H-Group beginner conventions.
        
        Implements:
        - Play Clues: Prioritize cluing immediately playable cards
        - 5 Save: Save 5s on chop with number 5 clue
        - 2 Save: Save 2s on chop with number 2 clue
        - Critical Save: Save last copy of cards on chop
        - Smart discarding: Prefer discarding from chop (oldest unclued)
        
        Args:
            node: The node to start rollouts from
            rollout_depth: Maximum number of moves per rollout
            num_rollouts: Number of rollouts to average (for variance reduction)
            use_parallel: Whether to run rollouts in parallel
            
        Returns:
            Tuple of (policy_distribution, value_estimate)
        """
        if use_parallel and num_rollouts > 1:
            parallel_rollout = ParallelConventionRollout(
                num_workers=min(num_rollouts, 4)
            )
            value, policy = parallel_rollout.run_rollouts(
                node.state, num_rollouts, rollout_depth
            )
        else:
            # Single or sequential rollouts
            policy_rollout = ConventionRolloutPolicy()
            
            if num_rollouts == 1:
                value, policy = policy_rollout.rollout(node.state, rollout_depth)
            else:
                # Multiple sequential rollouts
                scores = []
                policies = []
                for _ in range(num_rollouts):
                    score, prob = policy_rollout.rollout(node.state, rollout_depth)
                    scores.append(score)
                    policies.append(prob)
                
                value = float(np.mean(scores))
                policy = np.mean(policies, axis=0)
                if np.sum(policy) > 0:
                    policy = policy / np.sum(policy)
        
        return policy, value
    

    def run_parallel(self, remaining_time_ms: int = 0):
        if self.root_node is None:
            raise Exception("Root Node is None in MCTS run_parallel")
        
        children = self.root_node.expanded_children
        if not children:
            return

        pool_args = []
        for child in children:
            pool_args.append((
                child.state.copy(),
                child.last_action,
                child.c,
                child.prior,
                self.model,
                self.device,
                remaining_time_ms
            ))

        with multiprocessing.Pool(processes=6) as pool:
            results = pool.map(parallel_worker, pool_args)

        for child, (n_visits, value) in zip(children, results):
            child.n_visits += n_visits
            child.value += value
            
        total_visits = sum(r[0] for r in results)
        total_value = sum(r[1] for r in results)
        
        self.root_node.n_visits += total_visits
        self.root_node.value += total_value


    @torch.no_grad
    def run(self, use_rollouts: bool = False, run_parallel: bool = False,
            num_rollouts: int = 5, rollout_depth: int = 20,
            parallel_rollouts: bool = False):
        """
        Run MCTS search.
        
        Args:
            use_rollouts: If True, use convention-based rollouts instead of neural network
            run_parallel: If True, run tree parallelization (parallel MCTS trees)
            num_rollouts: Number of rollouts to average per node evaluation
            rollout_depth: Maximum depth for each rollout
            parallel_rollouts: If True, run rollouts in parallel (within each node)
        """

        if self.root_node is None:
            raise Exception("Root Node is None in MCTS run")
        
        if self.model is None and not use_rollouts:
            raise Exception("Model is None in MCTS run and rollouts are disabled")
        
        if use_rollouts:
            run_parallel = False
        
        policy = None
        value = None

        node = self.root_node
        start_time = time.time()
        while (time.time() - start_time) * 1000 < self.time_ms:

            if self.root_node.is_fully_expanded and run_parallel:
                self.run_parallel(self.time_ms - int((time.time() - start_time) * 1000))
                break

            if node.is_terminal:
                expanded = node
            else:
                expanded = self.expand(node)

            if expanded.is_terminal:
                result = expanded.state.score() / expanded.state.max_score()
                self.backprop(expanded, result)
                node = self.select()
                continue

            if use_rollouts:
                # Use convention-based rollout policy
                policy, value = self.rollout_policy(
                    expanded, 
                    rollout_depth=rollout_depth,
                    num_rollouts=num_rollouts,
                    use_parallel=parallel_rollouts
                )

            else:

                policy, value  = self.model(
                    torch.tensor(HLEGameState.encode_state(expanded.state, expanded.current_player), 
                                dtype=torch.float32, device=self.device).unsqueeze(0))
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy() # type: ignore
                value = value.item()

            expanded.set_prob_dist(policy)
            
            self.backprop(expanded, value)

            node = self.select()
        

        visit_counts = np.zeros(self.root_node.max_moves, dtype=np.float32)

        for child in self.root_node.expanded_children:
            if child is not None and child.last_action is not None:
                visit_counts[self.root_node.state.move_to_index(child.last_action)] = child.n_visits


        if np.sum(visit_counts) > 0:
            prob_dist_out = visit_counts / np.sum(visit_counts)
        else:
            print("Visit counts sum to zero in MCTS run, using valid moves uniform distribution")
            prob_dist_out = self.root_node.valid_moves_bool.astype(np.float32)
            prob_dist_out /= np.sum(prob_dist_out)
    
        return prob_dist_out, self.root_node.q_value


def parallel_worker(args):
    state, last_action, c, prior, model, device, time_ms = args
    root = Node(state, last_action, None, c, prior)
    mcts = MCTS(model, device, time_ms, root)
    root.mcts = mcts
    mcts.run()
    return root.n_visits, root.value

        



