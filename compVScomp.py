from steps.brain import get_best_move, print_board, make_move
import random

# Sample checkerboard matrix
sample_board = [
    [0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 2, 0, 2, 0, 2, 0],
    [0, 2, 0, 2, 0, 2, 0, 2],
    [2, 0, 2, 0, 2, 0, 2, 0]
]


player = random.choice([1, 2])

score1 = 0
score2 = 0

print(f"\rScore: 0 vs 0", end="")

while True:
    # Get the best move

    best_move = get_best_move(sample_board, player)

    if best_move is not None:

        if player == 1:
            player = 2
        elif player == 2:
            player = 1

        #print('Best move: ', best_move)

        sample_board = make_move(sample_board, best_move)

        ##print_board(sample_board)
    else:

        if player == 1:
            score1 += 1
        elif player == 2:
            score2 += 1
            
        print(f"\rScore: {score1} vs {score2}", end="")

        sample_board = [
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 2, 0, 2, 0, 2, 0],
            [0, 2, 0, 2, 0, 2, 0, 2],
            [2, 0, 2, 0, 2, 0, 2, 0]
        ]

        player = random.choice([1, 2])



    