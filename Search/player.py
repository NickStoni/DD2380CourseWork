#!/usr/bin/env python3

import math
import numpy as np
import time

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

TIME_LIMIT = 0.075
START_TIME = 0


class PlayerControllerHuman(PlayerController):

    def player_loop(self):
        """
            Function that generates the loop of the game. In each iteration
            the human plays through the keyboard and send
            this to the game through the sender. Then it receives an
            update of the game through receiver, with this it computes the
            next movement.
            :return:
            """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
            Main loop for the minimax next move search.
            :return:
            """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node):
        global START_TIME
        """
            Use minimax (and extensions) to find best possible next move for player 0 (green boat)
            :param initial_tree_node: Initial game tree node
            :type initial_tree_node: game_tree.Node
                (see the Node class in game_tree.py for more information!)
            :return: either "stay", "left", "right", "up" or "down"
            :rtype: str
            """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!

        START_TIME = time.time()
        depth = 1
        max_score = -math.inf
        best_move = 0

        memory = dict()
        move_ordering = dict()

        # Do Iterative deepening depth-first search and dynamic depth that is time-dependent
        while (time.time() - START_TIME) <= TIME_LIMIT * 0.82:
            score, move = self.alphabeta(node=initial_tree_node,
                                         a=-math.inf,
                                         b=math.inf,
                                         depth=depth,
                                         memory=memory,
                                         move_ordering=move_ordering)  # We will reuse memory across iterations

            if score > max_score:
                best_move = move
                max_score = score

            depth += 1

        return ACTION_TO_STR[best_move]

    def alphabeta(self, node, a, b, depth, memory, move_ordering):
        state = node.state
        player = state.get_player()

        hashable_state = HashableState(state)

        # Do not recompute states that have been encountered with equal or lower depth remaining
        if hashable_state in memory and memory[hashable_state][1] >= depth:
            return memory[hashable_state][0], node.move

        # Exiting condition is either the final state of the game or time limit or out of depth
        if depth == 0 or not state.get_fish_positions() or time.time() - START_TIME > 0.75 * TIME_LIMIT:
            if node.move is None:
                score = -math.inf if player == 0 else math.inf
                return score, -1

            return self.h(state), node.move

        if hashable_state in move_ordering:
            children = move_ordering[hashable_state]
        else:
            children = node.compute_and_get_children()
            children = [(children[i], i) for i in range(len(children))]

        # Green boat, a.k.a. MAX_SCORE
        if player == 0:
            v = -math.inf
            best_move = 0
            temp_children = []
            to_search = True

            for child in children:
                if to_search:
                    score, _ = self.alphabeta(node=child[0],
                                              a=a,
                                              b=b,
                                              depth=depth - 1,
                                              memory=memory,
                                              move_ordering=move_ordering)
                    temp_children.append((child[0], score))
                    if score > v:
                        v = score
                        best_move = child[0].move
                    a = max(a, v)
                    if b <= a:
                        to_search = False
                else:
                    temp_children.append((child[0], 10000000))
            temp_children = sorted(temp_children, key=lambda x: x[1])

        # Red boat, a.k.a. MIN_SCORE
        elif player == 1:
            v = math.inf
            best_move = 0
            temp_children = []
            to_search = True

            for child in children:
                if to_search:
                    score, _ = self.alphabeta(node=child[0],
                                              a=a,
                                              b=b,
                                              depth=depth - 1,
                                              memory=memory,
                                              move_ordering=move_ordering)
                    temp_children.append((child[0], score))
                    if score < v:
                        v = score
                        best_move = child[0].move
                    b = min(b, v)
                    if b <= a:
                        to_search = False
                else:
                    temp_children.append((child[0], -1000000))
            temp_children = sorted(temp_children, key=lambda x: -x[1])

        move_ordering[hashable_state] = temp_children
        if node.move is not None:
            memory[hashable_state] = (v, depth)  # Save the state, to avoid recomputation in the future

        return v, best_move

    def h(self, state):
        """
          Heuristic function to score each state.
          :param state (State class object)
          :return h_score (float)
        """

        fish_positions = state.get_fish_positions()
        fish_scores = state.get_fish_scores()
        player_positions = state.get_hook_positions()

        distance_and_score_h = 0
        dist_and_score = []

        for fish_index in fish_positions.keys():
            fish_position = fish_positions[fish_index]
            fish_score = fish_scores[fish_index]

            # Only consider fish that have not been caught and have positive score
            if fish_index != state.get_caught()[0] and fish_index != state.get_caught()[1] and fish_score > 0:
                dist_and_score.append((self.manhattan_dist(fish_position, player_positions[0]), fish_score))

        dist_and_score.sort(key=lambda x: x[0])  # Sort in ascending order

        # We are only interested in the closest 2 fish, proven to give better performance
        for i in range(min(len(dist_and_score), 2)):
            # This is normalized distance, we want to gain full points when dist is 0 and 0 points when dist
            # (max possible) is 30 (20 in y-dir and 10 in x-dir, due to endless plane)
            distance_and_score_h += dist_and_score[i][1] * (30 - dist_and_score[i][0]) / 30

        return self.score_h(state) + distance_and_score_h

    def score_h(self, state):
        fish_scores = state.get_fish_scores()

        caught_p0 = fish_scores[state.get_caught()[0]] if state.get_caught()[0] else 0
        caught_p1 = fish_scores[state.get_caught()[1]] if state.get_caught()[1] else 0

        # Calculate the difference in scores and the scores of the caught fish
        return state.player_scores[0] + caught_p0 - (state.player_scores[1] + caught_p1)

    def manhattan_dist(self, f, p):
        # The plane is endless, therefore the distance in x is the min distance from left and right direction
        return min(np.abs(p[0] - f[0]), np.abs(20 - max(p[0], f[0])) + min(f[0], p[0])) + np.abs(f[1] - p[1])


class HashableState:
    def __init__(self, state):
        self.state = state

    def __eq__(self, other):
        # Equality comparison between two objects
        return self.state.get_player() == other.state.get_player() \
               and self.state.get_player_scores() == other.state.get_player_scores() \
               and self.state.get_fish_positions() == other.state.get_fish_positions() \
               and self.state.get_hook_positions() == other.state.get_hook_positions()

    def __hash__(self):
        # This was proven to be one of the more efficient ways to compute hash
        return hash(tuple([self.state.get_player(),
                           frozenset(self.state.get_player_scores()),
                           frozenset(sorted(self.state.get_fish_positions().items())),
                           frozenset(sorted(self.state.get_hook_positions().items()))]))
