"""Demo script showing HLE game initialization, moves, and player observations."""

from rl_hanabi.game.hle_state import HLEGameState
from rl_hanabi.training.token_utils import GameConfig


def main():
    # Initialize a 3-player game with default options
    options = {
        "numSuits": 5,
        "numRanks": 5,
        "cardsPerHand": 5,
        "clueTokens": 8,
        "strikeTokens": 3,
        "startingPlayer": 0,
    }
    num_players = 3
    our_index = 0

    game_config = GameConfig(
        num_players=num_players,
        num_colors=options.get("numSuits", 5),
        num_ranks=options.get("numRanks", 5),
        hand_size=options.get("cardsPerHand", 5),
        max_information_tokens=options.get("clueTokens", 8),
        max_life_tokens=options.get("strikeTokens", 3),
        seed=options.get("seed", -1),
    )
    game_state = HLEGameState.from_table_options(
        game_config,
        starting_player=options.get("startingPlayer", 0),
    )

    print("=" * 60)
    print("Initial Game State")
    print("=" * 60)
    print(game_state)
    print()

    # Play some moves
    moves_to_play = 6  # Play 6 moves (2 full rounds)

    for turn in range(moves_to_play):
        if game_state.is_terminal():
            print("Game ended!")
            break

        current_player = game_state.current_player_index
        legal_moves = game_state.legal_moves()

        print("=" * 60)
        print(f"Turn {turn + 1}: Player {current_player}'s turn")
        print("=" * 60)

        # Print observation for the current player
        obs = game_state.observation_for_player(current_player)
        print(f"\n--- Player {current_player}'s Observation ---")

        print(f" FireWorks: {obs.fireworks()}")
        print(f" Discard Pile: {obs.discard_pile()}")
        for i, knowlegde in enumerate(obs.card_knowledge()):
            print(f" Card Knowledge {i} : {knowlegde}")
        for i, hand in enumerate(obs.observed_hands()):
            print(f" Player {i} Hand: {hand}")
        print(f" Last Moves: {obs.last_moves()}")
        print(f" Current Player Offset: {obs.cur_player_offset()}")


        # Pick specific moves to demonstrate different action types
        if legal_moves:
            if turn == 0:
                # Turn 1: Give a color clue to player 1
                from hanabi_learning_environment import pyhanabi
                move = pyhanabi.HanabiMove.get_reveal_color_move(1, 0)  # Clue Red to next player
                print(f"\nChosen move (Color Clue): {move}")
            elif turn == 1:
                # Turn 2: Give a rank clue to player 2
                move = pyhanabi.HanabiMove.get_reveal_rank_move(1, 0)  # Clue rank 1 to next player
                print(f"\nChosen move (Rank Clue): {move}")
            elif turn == 2:
                # Turn 3: Play a card
                move = pyhanabi.HanabiMove.get_play_move(0)  # Play card at index 0
                print(f"\nChosen move (Play): {move}")
            elif turn == 3:
                # Turn 4: Discard a card
                move = pyhanabi.HanabiMove.get_discard_move(4)  # Discard card at index 4
                print(f"\nChosen move (Discard): {move}")
            else:
                # Default: first legal move
                move = legal_moves[0]
                print(f"\nChosen move: {move}")
            game_state.apply_move(move)
        else:
            print("No legal moves available!")
            break

        print(f"\nScore: {game_state.score()}, "
              f"Clues: {game_state.clue_tokens}, "
              f"Strikes: {game_state.strikes}, "
              f"Deck: {game_state.deck_size()}")
        print()

    # Final state
    print("=" * 60)
    print("Final Game State")
    print("=" * 60)
    print(game_state)
    print(f"\nFinal Score: {game_state.score()}")


if __name__ == "__main__":
    main()
