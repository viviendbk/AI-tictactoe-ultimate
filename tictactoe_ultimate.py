import numpy as np
import cProfile, pstats, io
from numba import njit

# TODO : Create a machine learning algorithm to train the AI to determine the weights of the heuristic

# region Necessary variables
PLAYER = 0b00000000000000001000000000000000
MASKX = 0b000000001111111110000000000000000
MASKO = 0b000000000000000000000000111111111
transposition_table = {}

AVAILABLE_MOVES = {}
for number in range(2**9):
    bin_str = f"{number:09b}"
    AVAILABLE_MOVES[number] = sorted(
        8 - index for index, char in enumerate(bin_str) if char == "0"
    )

win_cons = [
    0b0000000000000111,  # Top row
    0b0000000000111000,  # Middle row
    0b0000000111000000,  # Bottom row
    0b0000000001001001,  # Left column
    0b0000000010010010,  # Middle column
    0b0000000100100100,  # Right column
    0b0000000100010001,  # Top-left to bottom-right diagonal
    0b0000000001010100,  # Top-right to bottom-left diagonal
]
# Define the winning line configurations
winning_lines = [
    [0, 1, 2],  # Top row
    [3, 4, 5],  # Middle row
    [6, 7, 8],  # Bottom row
    [0, 3, 6],  # Left column
    [1, 4, 7],  # Middle column
    [2, 5, 8],  # Right column
    [0, 4, 8],  # Top-left to bottom-right diagonal
    [2, 4, 6],  # Top-right to bottom-left diagonal
]
# endregion

# region Helper functions


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = "tottime"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def get_bit(bitmap, position):
    return (bitmap >> position) & 1


def get_row_bits(bitmap, pos1, pos2, pos3):
    return (
        (get_bit(bitmap, pos1) << 2)
        | (get_bit(bitmap, pos2) << 1)
        | get_bit(bitmap, pos3)
    )


def calculate_position(action):
    chosen_position = (
        action[1] + action[2] + 2 ** action[1]
        if (action[1] != 0)
        else action[1] + action[2]
    )
    return chosen_position


def is_player_X_turn(state_minimorp):
    """Return wether it's player X's turn or not.\n
    If state_bigmorp has to be passed, do state_bigmorp[0]"""
    return state_minimorp & PLAYER


def toggle_player(state_bigmorp):
    """Flip the player bit flag of all the minimorps in state_bigmorp."""
    for i in range(9):
        state_bigmorp[i] ^= PLAYER
    return state_bigmorp


def printRowX(state_minimorp, x):
    line_str = ""
    x_state_initial = bin(state_minimorp & MASKO)[2:]
    o_state_initial = bin((state_minimorp & MASKX) >> 16)[2:]
    x_state = ("".join("0" for x in range(9 - len(x_state_initial))) + x_state_initial)[
        ::-1
    ]
    o_state = ("".join("0" for x in range(9 - len(o_state_initial))) + o_state_initial)[
        ::-1
    ]
    for i in range(3):
        if x_state[3 * x + i] == "1" and o_state[3 * x + i] == "0":
            line_str += "X"
        if x_state[3 * x + i] == "0" and o_state[3 * x + i] == "0":
            line_str += "."
        if x_state[3 * x + i] == "0" and o_state[3 * x + i] == "1":
            line_str += "O"
        line_str += " "

    return "\t " + line_str


def display_board(state_bigmorp):
    for i in range(0, 9, 3):
        for j in range(3):
            print(
                printRowX(state_bigmorp[i], j)
                + printRowX(state_bigmorp[i + 1], j)
                + printRowX(state_bigmorp[i + 2], j)
            )
        print("")


# endregion

# region Heuristic helper functions


def check_continuable_win(bitmap, player):
    player_bits = bitmap if player == 1 else bitmap >> 16
    opponent_bits = bitmap if player == -1 else bitmap >> 16

    for line in winning_lines:
        count = sum(get_bit(player_bits, pos) == 1 for pos in line)
        opponent_count = sum(get_bit(opponent_bits, pos) == 1 for pos in line)
        if count == 2 and opponent_count == 0:
            return True

    return False


def has_blocked_opponent(bitmap, player):
    player_bits = bitmap if player == 1 else bitmap >> 16
    opponent_bits = bitmap if player == -1 else bitmap >> 16

    for line in winning_lines:
        count = sum(get_bit(player_bits, pos) == 1 for pos in line)
        opponent_count = sum(get_bit(opponent_bits, pos) == 1 for pos in line)
        if count == 1 and opponent_count == 2:
            return True

    return False


def opponent_has_won(state_minimorp, player):
    opponent = -player  # Determine the opponent player
    opponent_board = (
        state_minimorp if opponent == 1 else (state_minimorp >> 16)
    )  # Extract the opponent's board

    for win_con in win_cons:
        if (opponent_board & win_con) == win_con:
            return True

    return False


def double_winning_local_board(state_minimorp, player):
    # Check if both players can win in this local board
    opponent = -player
    player_can_win = check_continuable_win(
        state_minimorp, player
    ) and not has_blocked_opponent(state_minimorp, player)
    opponent_can_win = check_continuable_win(
        state_minimorp, opponent
    ) and not has_blocked_opponent(state_minimorp, opponent)

    if player_can_win and opponent_can_win:
        return True
    else:
        return False


# endregion


# region Minimorp functions
def results_minimorp(state_minimorp, action):
    chosen_position = (
        action[0] + action[1] + 2 ** action[0]
        if (action[0] != 0)
        else action[0] + action[1]
    )
    if is_player_X_turn(state_minimorp):
        state_minimorp = state_minimorp | 2**chosen_position
    else:
        state_minimorp = state_minimorp | (2**chosen_position << 16)
    return state_minimorp


def utility_minimorp(state_minimorp, is_player_X_turn):
    # If player X won (human)
    if is_player_X_turn == 1:
        if check_player_win(state_minimorp, 1):
            return 100
    else:
        if check_player_win(state_minimorp, -1):
            return 100
    return 0


def utility_minimorp2(state_minimorp):
    # Draw
    if ((state_minimorp | state_minimorp >> 16) & 0x1FF) == 0x1FF:
        return 2
    if check_player_win(state_minimorp, 1):
        return 1
    if check_player_win(state_minimorp, -1):
        return -1
    return 0


def is_terminal_minimorp(state_minimorp):
    if ((state_minimorp | state_minimorp >> 16) & 0x1FF) == 0x1FF:
        return True
    for i in range(len(win_cons)):
        if (state_minimorp & win_cons[i]) == win_cons[i]:
            return True
        elif (state_minimorp & (win_cons[i] << 16)) == (win_cons[i] << 16):
            return True
    return False


@njit
def check_player_win(bitmap, player):
    win_cons = [
        0b0000000000000111,  # ROW1
        0b0000000000111000,  # ROW 2
        0b0000000111000000,  # ROW 3
        0b0000000001001001,  # COL 1
        0b0000000010010010,  # COL 2
        0b0000000100100100,  # COL 3
        0b0000000100010001,  # DIAG 1
        0b0000000001010100,  # DIAG 2
    ]
    player_bits = (bitmap >> 16) if player == -1 else bitmap
    for win in win_cons:
        if (player_bits & win) == win:
            return True
    return False


# endregion


# region Bigmorp functions
def results_ultimate(state_bigmorp, action):
    new_state = state_bigmorp.copy()
    new_state[action[0]] = results_minimorp(
        new_state[action[0]], [action[1], action[2]]
    )
    return new_state


def actions_ultimate(state_bigmorp, constraint):
    all_actions = []
    if (
        constraint is not None
        and is_terminal_minimorp(state_bigmorp[constraint]) == False
    ):
        actions_in_minimorp = AVAILABLE_MOVES[
            ((state_bigmorp[constraint] | state_bigmorp[constraint] >> 16) & MASKO)
        ]
        all_actions.extend(
            [
                (constraint, 0, x)
                if (0 <= x <= 2)
                else (constraint, 1, x - 3)
                if (2 < x <= 5)
                else (constraint, 2, x - 6)
                for x in actions_in_minimorp
            ]
        )
    else:
        for i in range(9):
            # Si on peut jouer où on veut, et que le morpion n'est pas terminé, alors on liste les all_actions possibles
            if not is_terminal_minimorp(state_bigmorp[i]):
                available_moves = AVAILABLE_MOVES[
                    ((state_bigmorp[i] | state_bigmorp[i] >> 16) & MASKO)
                ]
                all_actions.extend(
                    [
                        (i, 0, x)
                        if (0 <= x <= 2)
                        else (i, 1, x - 3)
                        if (2 < x <= 5)
                        else (i, 2, x - 6)
                        for x in available_moves
                    ]
                )
    return all_actions


def utility_ultimate(state_bigmorp, player):
    result = []
    for i in range(9):
        result.append(float(utility_minimorp(state_bigmorp[i], player)))
    result = np.array(result).reshape((3, 3))

    sum_row = np.sum(result, axis=1)  # somme des valeurs de chaque ligne
    sum_col = np.sum(result, axis=0)  # somme des valeurs de chaque colonne

    sum_diag1 = np.trace(result)  # somme de la diagonale
    sum_diag2 = np.trace(np.rot90(result))  # somme de l'autre diagonale
    if player == 1:
        if (
            np.any(sum_row == 300)
            or np.any(sum_col == 300)
            or sum_diag1 == 300
            or sum_diag2 == 300
        ):
            return 1000
    elif player == -1:
        if (
            np.any(sum_row == 300)
            or np.any(sum_col == 300)
            or sum_diag1 == 300
            or sum_diag2 == 300
        ):
            return 1000
    return 0


def is_terminal_ultimate(state_bigmorp):
    is_full = np.all(
        [is_terminal_minimorp(morp) for morp in state_bigmorp]
    )  # check s'il n'y a pas state=0
    result = []
    for i in range(9):
        result.append(utility_minimorp(state_bigmorp[i], 0))
    result = np.array(result).reshape((3, 3))
    sum_row = np.sum(result, axis=1)  # somme des valeurs de chaque ligne
    sum_col = np.sum(result, axis=0)  # somme des valeurs de chaque colonne

    sum_diag1 = np.trace(result)  # somme de la diagonale
    sum_diag2 = np.trace(np.rot90(result))
    return (
        is_full
        or np.any(sum_row == 300)
        or np.any(sum_row == -300)
        or np.any(sum_col == 300)
        or np.any(sum_col == -300)
        or sum_diag1 in (300, -300)
        or sum_diag2 in (300, -300)
    )


def check_win_ultimate(state_bigmorp, player):
    if is_terminal_ultimate(state_bigmorp):
        display_board(state_bigmorp)
        value = utility_ultimate(state_bigmorp, player)
        if value == 0:
            print("Null")
        elif player == 1:
            print("You won!")
        elif player == -1:
            print("You lost")
        return True
    return False


def update_bitmap_bigmorp_real(state_bigmorp, to_be_updated, state_bigmorp_real=0):
    bitmap = state_bigmorp_real
    if to_be_updated == -1:
        for i in range(9):
            value = utility_minimorp2(state_bigmorp[i])
            if value == 1:
                bitmap |= 2**i
            elif value == -1:
                bitmap |= 2 ** (i + 16)
            elif value == 2:
                bitmap |= 2**i
                bitmap |= 2 ** (i + 16)
    else:
        value = utility_minimorp2(state_bigmorp[to_be_updated])
        if value == 1:
            bitmap |= 2**to_be_updated
        elif value == -1:
            bitmap |= 2 ** (to_be_updated + 16)
        elif value == 2:
            bitmap |= 2**to_be_updated
            bitmap |= 2 ** (to_be_updated + 16)
    return bitmap


# endregion


# region Game
def human_round(state_bigmorp, constraint, round):
    if constraint is None or is_terminal_minimorp(state_bigmorp[constraint]):
        while True:
            coord_player = input("Your move (ex: 1,2,3): ")
            coord_player = coord_player.replace(" ", "")
            if coord_player.count(",") == 2:
                coords = coord_player.split(",")
                if all(coord.isdigit() and 0 <= int(coord) <= 8 for coord in coords):
                    coord_player = list(map(int, coords))
                    break
            print("Invalid input. Please enter a valid move.")
    else:
        while True:
            print("Round n°", round)
            coord_player = input(f"Your move in tic tac toe n°{constraint} (ex: 1,2): ")
            coord_player = coord_player.replace(" ", "")
            if coord_player.count(",") == 1:
                coords = coord_player.split(",")
                if all(coord.isdigit() and 0 <= int(coord) <= 2 for coord in coords):
                    coord_player = [constraint] + list(map(int, coords))
                    break
            print("Invalid input. Please enter a valid move.")
    state_bigmorp = results_ultimate(state_bigmorp, coord_player)
    constraint = calculate_position(coord_player)
    return state_bigmorp, constraint


def AI_round(state_bigmorp, constraint, round):
    best_action, best_score = minimax_decision_ultimate(
        state_bigmorp, 3, round, constraint
    )
    state_bigmorp = results_ultimate(state_bigmorp, best_action)
    constraint = calculate_position(best_action)
    return state_bigmorp, constraint


def ultimate_tictactoe(starting_player):
    # Human start
    if starting_player == 1:
        empty_state = 0b0000_0000_0000_0000_0000_0000_0000_0000
    else:
        empty_state = 0b0000_0000_0000_0000_1000_0000_0000_0000
    state_bigmorp = [
        empty_state,
        empty_state,
        empty_state,
        empty_state,
        empty_state,
        empty_state,
        empty_state,
        empty_state,
        empty_state,
    ]
    constraint = None
    round = 1
    while True:
        toggle_player(state_bigmorp)
        display_board(state_bigmorp)
        if starting_player == 1:
            state_bigmorp, constraint = human_round(state_bigmorp, constraint, round)
            if check_win_ultimate(state_bigmorp, 1):
                break
            display_board(state_bigmorp)
            print("IA playing...")
            state_bigmorp = toggle_player(state_bigmorp)
            state_bigmorp, constraint = AI_round(state_bigmorp, constraint, round)
            if check_win_ultimate(state_bigmorp, -1):
                break
        else:
            print("IA playing...")
            state_bigmorp, constraint = AI_round(state_bigmorp, constraint, round)
            if check_win_ultimate(state_bigmorp, -1):
                break
            display_board(state_bigmorp)
            state_bigmorp = toggle_player(state_bigmorp)
            state_bigmorp, constraint = human_round(state_bigmorp, constraint, round)
            if check_win_ultimate(state_bigmorp, 1):
                break
        round += 1


# endregion


# region AI Battle
def AI_battle():
    empty_state = 0b0000_0000_0000_0000_0000_0000_0000_0000
    state_bigmorp = [
        empty_state,
        empty_state,
        empty_state,
        empty_state,
        empty_state,
        empty_state,
        empty_state,
        empty_state,
        empty_state,
    ]
    constraint = None
    round = 1
    while True:
        state_bigmorp = toggle_player(state_bigmorp)
        best_action_AI_1, best_score = minimax_decision_ultimate(
            state_bigmorp, 3, round, constraint
        )
        state_bigmorp = results_ultimate(state_bigmorp, best_action_AI_1)
        constraint = calculate_position(best_action_AI_1)
        display_board(state_bigmorp)
        if check_win_ultimate(state_bigmorp, 1):
            break
        print("IA n°1 playing...")

        best_action_AI_2, best_score = minimax_decision_ultimate_AI_2(
            state_bigmorp, 3, round, constraint
        )
        state_bigmorp = results_ultimate(state_bigmorp, best_action_AI_2)
        constraint = calculate_position(best_action_AI_2)
        display_board(state_bigmorp)
        if check_win_ultimate(state_bigmorp, -1):
            break
        print("IA n°2 playing...")
        round += 1


def heuristic_minimorp_AI2(state_minimorp, player, round):
    if state_minimorp in transposition_table:
        return transposition_table[state_minimorp]
    sum_eval = 0
    play_middle = 0
    did_block = 0
    create_pairs = 0
    has_won = 0
    opponent_won = 0
    # Encourage playing in the middle - add 2 points
    if player == 1:
        if get_bit(state_minimorp, 4) == 1:
            play_middle = 2
    elif player == -1:
        if get_bit(state_minimorp, 20) == 1:
            play_middle = 2
    # Block pairs in minimorp - add 10 points
    if has_blocked_opponent(state_minimorp, player):
        did_block = 10
    # Create pairs - add 5 points
    if check_continuable_win(state_minimorp, player):
        create_pairs = 5
    # Encourage to win in minimorp - add 100 points
    has_won = utility_minimorp(state_minimorp, player)
    # Discourage to play on a position that will make the other player win - add -20 points
    if utility_minimorp(state_minimorp, -player) == 100:
        opponent_won = -20
    # In the beginning of the game we have to encourage to make pairs and block pairs
    if 1 <= round <= 10:
        play_middle *= 1.5
        did_block *= 2
        create_pairs *= 3
        has_won *= 0.5
        opponent_won *= 1
    # In the middle of the game we have to encourage to win
    elif 10 < round <= 15:
        play_middle *= 2
        did_block *= 2
        create_pairs *= 1
        has_won *= 1.5
        opponent_won *= 1.5
    # In the end of the game we have to encourage to win
    else:
        play_middle *= 3
        did_block *= 5
        create_pairs *= 1
        has_won *= 2
        opponent_won *= 1.5
    sum_eval = play_middle + did_block + create_pairs + has_won + opponent_won
    transposition_table[state_minimorp] = sum_eval
    return sum_eval


def heuristic_AI2(state_bigmorp, action, is_player_X_turn, round):
    if action == None:
        return 0
    sum_eval = 0
    # Bouger state_bigmorp_real en paramètre pour être mis dans le minimax_value_ultimate
    state_bigmorp_real = update_bitmap_bigmorp_real(state_bigmorp, -1)
    player = 1 if is_player_X_turn else -1
    for i in range(9):
        sum_eval += (
            heuristic_minimorp_AI2(state_bigmorp[i], player, round) * 1.5
            if (i == action[0])
            else heuristic_minimorp_AI2(state_bigmorp[i], player, round)
        )

    middle_win = 0
    middle_play_if_will_not_be_won = 0
    create_pair = 0
    did_block = 0
    dont_play_in_full_board = 0
    # Encourage to win in bigmorp - add 100 points
    bigmorp_win = utility_minimorp(state_bigmorp_real, player) * 5

    # Discourage to play on a position that will make the other player win - add -100 points
    bigmorp_lose = utility_minimorp(state_bigmorp_real, -player) * 5

    # Discourage to play in a full board - add -10 points
    if is_terminal_minimorp(state_bigmorp[action[0]]):
        dont_play_in_full_board -= 10
    # Encourage to win in the middle board - add 50 points
    if action[0] == 4 and utility_minimorp(state_bigmorp_real, player) == 100:
        middle_win = 50

    # Encourage to play in the center board - add 10 points [4]
    if action[0] == 4 and utility_minimorp(state_bigmorp_real, -player) == 0:
        middle_play_if_will_not_be_won = 10

    # Create pairs - add 5 points
    if check_continuable_win(state_bigmorp_real, player) and not has_blocked_opponent(
        state_bigmorp_real, -player
    ):
        create_pair += 5

    # Block pairs in bigmorp - add 10 points
    if has_blocked_opponent(state_bigmorp_real, player):
        did_block += 10

    # In the beginning of the game we have to encourage to make pairs and block pairs
    if 1 <= round <= 10:
        middle_play_if_will_not_be_won *= 2
        middle_win *= 3
        create_pair *= 1.5
        did_block *= 2
        dont_play_in_full_board *= 1
    elif 10 < round <= 15:
        bigmorp_win *= 10
        middle_win *= 3
        create_pair *= 1
        did_block *= 5
        middle_play_if_will_not_be_won *= 2
        dont_play_in_full_board *= 4
    else:
        bigmorp_win *= 100
        middle_win *= 4
        create_pair *= 1
        did_block *= 0
        middle_play_if_will_not_be_won *= 10
        dont_play_in_full_board *= 3

    sum_eval += (
        bigmorp_win
        + bigmorp_lose
        + middle_win
        + middle_play_if_will_not_be_won
        + create_pair
        + did_block
        + dont_play_in_full_board
    )
    return sum_eval * player


def minimax_value_ultimate_AI_2(
    state_bigmorp, constraint, previous_action, alpha, beta, round, depth=0
):
    if is_terminal_ultimate(state_bigmorp) or depth == 0:
        return heuristic_AI2(
            state_bigmorp, previous_action, is_player_X_turn(state_bigmorp[0]), round
        )

    if is_player_X_turn(state_bigmorp[0]):
        maxEval = -100000
        for action in actions_ultimate(state_bigmorp, constraint):
            # Create the new state with the action applied to it
            new_state_bigmorp = results_ultimate(state_bigmorp, action)
            # Calculate the new constraint
            new_constraint = calculate_position(action)
            # Switch the player bit flag - Player -1 turn
            new_state_bigmorp = toggle_player(new_state_bigmorp)
            currentEval = minimax_value_ultimate_AI_2(
                new_state_bigmorp, new_constraint, action, alpha, beta, round, depth - 1
            )
            maxEval = max(maxEval, currentEval)
            alpha = max(alpha, currentEval)
            if beta <= alpha:
                break
        return maxEval

    else:
        minEval = 100000
        for action in actions_ultimate(state_bigmorp, constraint):
            # Create the new state with the action applied to it
            new_state_bigmorp = results_ultimate(state_bigmorp, action)
            # Calculate the new constraint
            new_constraint = calculate_position(action)
            # Switch the player bit flag - Player 1 turn
            new_state_bigmorp = toggle_player(new_state_bigmorp)
            currentEval = minimax_value_ultimate_AI_2(
                new_state_bigmorp, new_constraint, action, alpha, beta, round, depth - 1
            )
            minEval = min(minEval, currentEval)
            beta = min(beta, currentEval)
            if beta <= alpha:
                break
        return minEval


def minimax_decision_ultimate_AI_2(state_bigmorp, max_depth, round, constraint=None):
    best_action = None
    best_value = float("inf")

    for action in actions_ultimate(state_bigmorp, constraint):
        # Apply the action to create a new state
        new_state_bigmorp = results_ultimate(state_bigmorp, action)
        new_constraint = calculate_position(action)
        new_state_bigmorp = toggle_player(new_state_bigmorp)

        value = minimax_value_ultimate_AI_2(
            new_state_bigmorp, new_constraint, None, -10000, 10000, round, max_depth
        )

        # display_board(new_state_bigmorp)
        # print(f"Action: {action} - Value: {value}")

        if value < best_value:
            best_value = value
            best_action = action
        elif value == best_value:
            if np.random.randint(2) == 0:
                best_value = value
                best_action = action
    # Return the best action and value
    return best_action, best_value


# endregion


# def heuristic_simplified(state_bigmorp, action, is_player_X_turn, round):
#     if action is None:
#         return 0

#     player = 1 if is_player_X_turn else -1
#     state_bigmorp_real = update_bitmap_bigmorp_real(state_bigmorp, -1)

#     local_boards_score = 0
#     global_board_score = 0

#     for i in range(9):
#         local_board_score = heuristic_minimorp(state_bigmorp[i], player, round)
#         if i == action[0]:
#             local_board_score *= 1.5
#         local_boards_score += local_board_score

#     global_board_score = heuristic_minimorp(state_bigmorp_real, player, round)

#     # Balance the weights between local boards and global board
#     total_score = local_boards_score * 0.8 + global_board_score * 1.2
#     return total_score * player


def heuristic_minimorp(state_minimorp, player, round):
    # if state_minimorp in transposition_table:
    #     return transposition_table[state_minimorp]

    score = 0
    bonus_play_middle = 0
    bonus_did_block = 0
    bonus_create_pairs = 0
    bonus_win = 0
    penalty_opponent_win = 0
    penalty_double_winning_board = 0

    # Bonus for playing in the middle
    if player == 1:
        if get_bit(state_minimorp, 4) == 1:
            bonus_play_middle = 2
    elif player == -1:
        if get_bit(state_minimorp, 20) == 1:
            bonus_play_middle = 2

    # Bonus for blocking pairs
    if has_blocked_opponent(state_minimorp, player):
        bonus_did_block = 5

    # Bonus for creating pairs
    if check_continuable_win(state_minimorp, player) and not has_blocked_opponent(
        state_minimorp, player
    ):
        bonus_create_pairs = 5

    # Bonus for winning in minimorp
    bonus_win = utility_minimorp(state_minimorp, player) * 0.25

    # Penalty for playing on a position that will make the other player win
    if utility_minimorp(state_minimorp, -player) == 100:
        penalty_opponent_win = -20

    # Discourage to play on a board with each player can win and the opponent play first
    # if double_winning_local_board(state_minimorp, player):
    #     penalty_double_winning_board = -50
    # Weight adjustments based on round number
    weights = [[1.5, 2, 3, 1, 1, 1], [2, 2, 1, 1.5, 1.5, 2], [3, 2, 1, 2, 1.5, 3]]

    weight_index = 0
    if 1 <= round <= 5:
        weight_index = 0
    elif 5 < round <= 15:
        weight_index = 1
    else:
        weight_index = 2

    # Calculate total score
    bonus_play_middle *= weights[weight_index][0]
    bonus_did_block *= weights[weight_index][1]
    bonus_create_pairs *= weights[weight_index][2]
    bonus_win *= weights[weight_index][3]
    penalty_opponent_win *= weights[weight_index][4]
    # penalty_double_winning_board *= weights[weight_index][5]

    score = (
        bonus_play_middle
        + bonus_did_block
        + bonus_create_pairs
        + bonus_win
        + penalty_opponent_win
        + penalty_double_winning_board
    )

    # transposition_table[state_minimorp] = score
    return score


def heuristic(state_bigmorp, action, player, round):
    if action == None:
        return 0
    sum_eval = 0
    # Bouger state_bigmorp_real en paramètre pour être mis dans le minimax_value_ultimate
    state_bigmorp_real = update_bitmap_bigmorp_real(state_bigmorp, -1)
    player = 1 if player else -1
    for i in range(9):
        sum_eval += (
            heuristic_minimorp(state_bigmorp[i], player, round) * 1
            if (i == action[0])
            else heuristic_minimorp(state_bigmorp[i], player, round)
        )

    middle_win = 0
    middle_play_if_will_not_be_won = 0
    create_pair = 0
    did_block = 0
    block_in_mid = 0
    dont_play_in_full_board = 0
    dont_make_him_win = 0
    dont_play_opponent_block_us = 0
    win_corner_board = 0
    # Encourage to win in bigmorp - add 100 points
    bigmorp_win = utility_minimorp(state_bigmorp_real, player)

    # Discourage to play on a position that will make the other player win - add -100 points
    bigmorp_lose = -utility_minimorp(state_bigmorp_real, -player)

    # Discourage to play a move that will make the other player win a mini board - add -20 points
    next_minimorp = action[1] * 3 + action[2]
    if opponent_has_won(state_bigmorp[next_minimorp], player):
        dont_make_him_win = -30
    # Discourage to play a move that will lead the other player to block us - add -20 points
    if has_blocked_opponent(state_bigmorp[next_minimorp], player):
        dont_play_opponent_block_us = -30
    # Encourage to win in the middle board - add 50 points
    if player == 1:
        if action[0] == 4 and get_bit(state_bigmorp_real, 4) == 1:
            middle_win = 30
    elif player == -1:
        if action[0] == 4 and get_bit(state_bigmorp_real, 20) == 1:
            middle_win = 30

    # Encourage to play in the center board - add 10 points [4]
    if player == 1:
        if action[0] == 4 and get_bit(state_bigmorp_real, 4) == 0:
            middle_play_if_will_not_be_won = 10
    elif player == -1:
        if action[0] == 4 and get_bit(state_bigmorp_real, 20) == 0:
            middle_play_if_will_not_be_won = 10

    # Block in the middle
    if action[0] == 4 and has_blocked_opponent(state_bigmorp_real, player):
        block_in_mid = 30

    # Create pairs - add 5 points
    if check_continuable_win(state_bigmorp_real, player) and not has_blocked_opponent(
        state_bigmorp_real, player
    ):
        create_pair += 10

    # Block pairs in bigmorp - add 10 points
    if has_blocked_opponent(state_bigmorp_real, player):
        did_block += 10

    # Discourage to play in a full board - add -10 points
    if is_terminal_minimorp(state_bigmorp[action[0]]):
        dont_play_in_full_board -= 5

    # In the beginning of the game we have to encourage to make pairs and block pairs
    if 1 <= round <= 5:
        middle_win *= 3
        create_pair *= 3
        did_block *= 2
        block_in_mid *= 3
        middle_play_if_will_not_be_won *= 2
        dont_play_in_full_board *= 1
        dont_make_him_win *= 1
        dont_play_opponent_block_us *= 1
        # win_corner_board *= 2
    elif 5 < round <= 15:
        bigmorp_win *= 10
        bigmorp_lose *= 10
        middle_win *= 3
        create_pair *= 1
        did_block *= 5
        block_in_mid *= 3
        middle_play_if_will_not_be_won *= 2
        dont_play_in_full_board *= 4
        dont_make_him_win *= 1.5
        dont_play_opponent_block_us *= 1
        # win_corner_board *= 1
    else:
        bigmorp_win *= 1000
        bigmorp_lose *= 100
        middle_win *= 4
        create_pair *= 1
        block_in_mid *= 5
        did_block *= 100
        middle_play_if_will_not_be_won *= 10
        dont_play_in_full_board *= 3
        dont_make_him_win *= 1.5
        dont_play_opponent_block_us *= 1
        # win_corner_board *= 1

    sum_eval += (
        bigmorp_win
        + bigmorp_lose
        + middle_win
        + middle_play_if_will_not_be_won
        + create_pair
        + did_block
        + dont_play_in_full_board
        + dont_make_him_win
        + dont_play_opponent_block_us
    )
    return sum_eval * player


def minimax_value_ultimate(
    state_bigmorp, constraint, previous_action, alpha, beta, round, depth=0
):
    if is_terminal_ultimate(state_bigmorp) or depth == 0:
        return heuristic(
            state_bigmorp, previous_action, is_player_X_turn(state_bigmorp[0]), round
        )

    if is_player_X_turn(state_bigmorp[0]):
        maxEval = -float("inf")
        for action in actions_ultimate(state_bigmorp, constraint):
            # Create the new state with the action applied to it
            new_state_bigmorp = results_ultimate(state_bigmorp, action)
            # Calculate the new constraint
            new_constraint = calculate_position(action)
            # Switch the player bit flag - Player -1 turn
            new_state_bigmorp = toggle_player(new_state_bigmorp)
            currentEval = minimax_value_ultimate(
                new_state_bigmorp,
                new_constraint,
                action,
                alpha,
                beta,
                round,
                depth - 1,
            )
            maxEval = max(maxEval, currentEval)
            alpha = max(alpha, currentEval)
            if beta <= alpha:
                break
        return maxEval

    else:
        minEval = float("inf")
        for action in actions_ultimate(state_bigmorp, constraint):
            # Create the new state with the action applied to it
            new_state_bigmorp = results_ultimate(state_bigmorp, action)
            # Calculate the new constraint
            new_constraint = calculate_position(action)
            # Switch the player bit flag - Player 1 turn
            new_state_bigmorp = toggle_player(new_state_bigmorp)
            currentEval = minimax_value_ultimate(
                new_state_bigmorp,
                new_constraint,
                action,
                alpha,
                beta,
                round,
                depth - 1,
            )
            minEval = min(minEval, currentEval)
            beta = min(beta, currentEval)
            if beta <= alpha:
                break
        return minEval


def minimax_decision_ultimate(state_bigmorp, max_depth, round, constraint=None):
    best_action = None
    best_value = float("inf")

    for action in actions_ultimate(state_bigmorp, constraint):
        # Apply the action to create a new state
        new_state_bigmorp = results_ultimate(state_bigmorp, action)
        new_constraint = calculate_position(action)
        new_state_bigmorp = toggle_player(new_state_bigmorp)

        value = minimax_value_ultimate(
            new_state_bigmorp,
            new_constraint,
            None,
            -10000,
            10000,
            round,
            max_depth,
        )

        display_board(new_state_bigmorp)
        print(f"Action: {action} - Value: {value}")

        if value < best_value:
            best_value = value
            best_action = action
        # elif value == best_value:
        #     if np.random.randint(2) == 0:
        #         best_value = value
        #         best_action = action
    # Return the best action and value
    return best_action, best_value


# Testing the functions
def test_functions():
    bitmap = 0b0000_0000_0010_0001_0000_0000_0000_0101

    # Test get_bit
    print(get_bit(bitmap, 0))  # Output: 1
    print(get_bit(bitmap, 1))  # Output: 0
    print(get_bit(bitmap, 16))  # Output: 1
    print(get_bit(bitmap, 17))  # Output: 0
    print("--------------------")
    # Test get_row_bits
    print(get_row_bits(bitmap, 6, 3, 0))  # Output: 5 (0b101)
    print(get_row_bits(bitmap, 1, 2, 3))  # Output: 1 (0b001)
    print(get_row_bits(bitmap, 6, 7, 8))  # Output: 5 (0b101)
    print("--------------------")
    # Test check_player_win
    print(check_player_win(bitmap, 1))  # Output: False
    print(check_player_win(bitmap, -1))  # Output: False
    print("--------------------")
    bitmap = 0b0000_0000_0001_0000_0000_0000_0010_0111
    print(check_player_win(bitmap, 1))  # Output: True
    print(check_player_win(bitmap, -1))  # Output: False
    print("--------------------")
    # Test check_continuable_win
    print(check_continuable_win(bitmap, 1))  # Output: False
    print(check_continuable_win(bitmap, -1))  # Output: False
    print("--------------------")
    bitmap = 0b0000_0000_0000_0000_0000_0000_0001_0001
    print(check_continuable_win(bitmap, 1))  # Output: True
    print(check_continuable_win(bitmap, -1))  # Output: False
    print("--------------------")
    # Diagonal
    bitmap = 0b0000_0001_0001_0000_0000_0000_0000_0001
    print(has_blocked_opponent(bitmap, 1))  # Output: True
    # Anti-diagonal
    bitmap = 0b0000_0000_0000_0100_0000_0000_0101_0000
    print(has_blocked_opponent(bitmap, -1))  # Output: True


# Run the test functions
test_functions()


def main():
    starting_player = int(input("Enter the starting player (1: human/-1: IA): "))
    ultimate_tictactoe(starting_player)
    # AI_battle()
    state_bigmorp = [
        0b0000_0000_1001_0010_0000_0000_0000_0000,
        0b0000_0001_0101_0100_0000_0000_0000_0011,
        0b0000_0001_1001_0000_0000_0000_0000_0001,
        0b0000_0000_0000_0000_0000_0000_0000_0000,
        0b0000_0000_0001_0000_0000_0000_0000_0111,
        0b0000_0000_0000_0000_0000_0000_0000_0000,
        0b0000_0000_0000_0000_0000_0000_0000_0001,
        0b0000_0000_0000_0000_0000_0000_0000_0011,
        0b0000_0000_0000_0000_0000_0000_0000_0011,
    ]


main()
