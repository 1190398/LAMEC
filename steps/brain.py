import random

## UPPER PART OF THE MATRIX IS ALWAYS 1
## LOWER PART OF THE MATRIX IS ALWAYS 0

##THIS IMPLEMENTS THE CHECKERS GAME
    ## DOES: KINGS; MULTI EATING

# Constants for players
PLAYER_1 = 1
PLAYER_2 = 2
KING_1 = 3
KING_2 = 4

# Example of setting global variables
search_depth = 6
mandatory_eating = True  # Set to True to make eating a piece mandatory

def evaluate(board, current_player):
    score_player_1 = 0
    score_player_2 = 0

    eatingScore = 1
    kingScore = 2

    for row in board:
        for cell in row:
            if cell == PLAYER_1:
                score_player_1 += eatingScore
            elif cell == KING_1:
                score_player_1 += kingScore
            elif cell == PLAYER_2:
                score_player_2 += eatingScore
            elif cell == KING_2:
                score_player_2 += kingScore

    if current_player == PLAYER_2:
    # Reverse scores if it's Player 2's turn
        score_player_1, score_player_2 = score_player_2, score_player_1

    return score_player_1 - score_player_2

def get_regular_moves(board, player, row, col):
    moves = []
    directions = []

    if board[row][col] == PLAYER_1:
        directions = [(1, -1), (1, 1)]
    elif board[row][col] == PLAYER_2:
        directions = [(-1, -1), (-1, 1)]
    else:
        directions = [(1, -1), (1, 1), (-1, -1), (-1, 1)]

    for dir_row, dir_col in directions:
        new_row, new_col = row + dir_row, col + dir_col
        if 0 <= new_row < 8 and 0 <= new_col < 8 and board[new_row][new_col] == 0:
            moves.append(((row, col), (new_row, new_col)))

    return moves      

def get_eating_moves(board, player, row, col):
    moves = []
    directions = []

    if board[row][col] == PLAYER_1:
        directions = [(2, -2), (2, 2)]
    elif board[row][col] == PLAYER_2:
        directions = [(-2, -2), (-2, 2)]
    else:
        directions = [(2, -2), (2, 2), (-2, -2), (-2, 2)]

    for dir_row, dir_col in directions:
        new_row, new_col = row + dir_row, col + dir_col
        mid_row, mid_col = row + dir_row // 2, col + dir_col // 2

        if 0 <= new_row < 8 and 0 <= new_col < 8 and board[new_row][new_col] == 0 and 0 <= mid_row < 8 and 0 <= mid_col < 8 and board[mid_row][mid_col] != 0 and board[mid_row][mid_col] != board[row][col] and board[mid_row][mid_col] != board[row][col]-2 and board[mid_row][mid_col] != board[row][col]+2:
            moves.append(((row, col), (new_row, new_col)))

    return moves

def get_recursive_eating_moves(board, player, row, col):
    moves = []

    eating_moves = get_eating_moves(board, player, row, col)
    for move in eating_moves:
        new_matrix = make_move(board, move)
        recursive_moves = get_recursive_eating_moves(new_matrix, player, move[1][0], move[1][1])

        # Flatten the nested tuple structure
        if recursive_moves:
            moves.append((move,) + recursive_moves[0])
        else:
            moves.append((move,))

    return moves

def get_multi_moves(board, player, row, col):
    moves = []
    recursive_moves = get_recursive_eating_moves(board, player, row, col)
    for move in recursive_moves:
        updated_move = []
        updated_move.append(move[0][0])
        for submove in move:
            updated_move.append(submove[1])
        moves.append(tuple(updated_move))

    if mandatory_eating == True:
        return moves
    else:
        expanded_moves = []
        for move in moves:
            combinations = [tuple(move[:i+1]) for i in range(len(move))]
            for index in range(len(combinations)-1):
                expanded_moves.append(combinations[index+1])
        return expanded_moves

def get_possible_moves(board, player):
    moves = []
    regular_moves = []
    eating_moves = []

    for i in range(8):
        for j in range(8):
            if board[i][j] == player or board[i][j] == player + 2:
                regular_moves += get_regular_moves(board, player, i, j)
                eating_moves += get_multi_moves(board, player, i, j)

    if mandatory_eating == True:
        if eating_moves != []:
            moves = eating_moves
        else:
            moves = regular_moves
    else:
        moves = eating_moves
        moves.extend(regular_moves)

    # Sort moves based on their evaluation values
    moves.sort(key=lambda move: evaluate(make_move(board, move), player), reverse=True)

    return moves

def make_move(board, move):
    # Create a new board with the given move applied
    new_board = [row.copy() for row in board]

    for index in range(len(move) - 1):
        start = move[index]
        end = move[index + 1]

        # Move the piece to the new position
        new_board[end[0]][end[1]] = new_board[start[0]][start[1]]
        new_board[start[0]][start[1]] = 0

        # Check if the move is an eating move
        is_eating_move = abs(end[0] - start[0]) == 2

        # If it's an eating move, remove the captured piece
        if is_eating_move:
            mid_row, mid_col = (start[0] + end[0]) // 2, (start[1] + end[1]) // 2
            new_board[mid_row][mid_col] = 0

        # Check for king promotion
        if end[0] == 7 and new_board[end[0]][end[1]] == PLAYER_1:
            new_board[end[0]][end[1]] = KING_1
        elif end[0] == 0 and new_board[end[0]][end[1]] == PLAYER_2:
            new_board[end[0]][end[1]] = KING_2



    return new_board



def minimax(board, depth, alpha, beta, maximizing_player, current_player):
    if depth == 0 or not any(PLAYER_1 in row or PLAYER_2 in row for row in board):
        return evaluate(board, current_player)

    if maximizing_player:
        max_eval = float('-inf')
        for move in get_possible_moves(board, current_player):
            new_board = make_move(board, move)
            eval = minimax(new_board, depth - 1, alpha, beta, False, current_player)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:
        min_eval = float('inf')
        for move in get_possible_moves(board, 3 - current_player):
            new_board = make_move(board, move)
            eval = minimax(new_board, depth - 1, alpha, beta, True, current_player)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval


def get_best_move(board, current_player):
    best_moves = []
    best_val = float('-inf')
    alpha = float('-inf')
    beta = float('inf')

    for move in get_possible_moves(board, current_player):
        new_board = make_move(board, move)
        move_val = minimax(new_board, search_depth - 1, alpha, beta, False, current_player)

        if move_val > best_val:
            best_val = move_val
            best_moves = [move]
        elif move_val == best_val:
            best_moves.append(move)

        #print(f"Move: {move}, Evaluation Value: {move_val}")

    return random.choice(best_moves) if best_moves else None



def print_board(board):
    for row in board:
        print(" ".join(str(cell) for cell in row))
