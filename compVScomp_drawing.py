# main.py
import tkinter as tk
from util.drawBoard import create_board, redraw_board, run_mainloop
from steps.brain import get_best_move, make_move, print_board
import random

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

root = tk.Tk()
board = create_board(root)

# Initialize the board with the sample matrix
redraw_board(board, sample_board)

playDelay = 300

player = random.choice([1, 2])
score1 = 0
score2 = 0

def play():
    global player, sample_board, score1, score2

    # Get the best move
    best_move = get_best_move(sample_board, player)

    if best_move is not None:
        if player == 1:
            player = 2
        elif player == 2:
            player = 1

        sample_board = make_move(sample_board, best_move)

        # Update the board with the new matrix
        redraw_board(board, sample_board)

        # Schedule the play function again after 0.5 seconds
        root.after(playDelay, play)
    else:
        if player == 1:
            score1 += 1
        elif player == 2:
            score2 += 1

        # Print the final score
        print(f"\rScore: {score1} vs {score2}", end="")

# Function to start the game when Enter key is pressed
def start_game(event):
    global sample_board
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
    play()

# Bind the start_game function to the Enter key
root.bind("<Return>", start_game)

# Run the main loop
run_mainloop(root)
