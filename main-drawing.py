# main.py
import tkinter as tk
from util.drawBoard import create_board, redraw_board, run_mainloop
from steps.brain import get_best_move, make_move, print_board
from steps.serial import choose_serial_port, initialize_serial_connection, wait_for_command, build_move_string, send_move
from steps.recon import main_recon
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

root = tk.Tk()
board = create_board(root)

# Initialize the board with the sample matrix
redraw_board(board, sample_board)

# Choose the COM port dynamically
chosen_port = choose_serial_port()

# Initialize the serial connection
serial_connection = initialize_serial_connection(chosen_port)

player = 2
score1 = 0
score2 = 0

def play():
    global player, sample_board, score1, score2

    best_move = None

    matrix = main_recon()

    if player == 1:  # robot
        # Call the function to wait for the "start" signal
        wait_for_command(serial_connection, "/start")

        # Get the best move
        best_move = get_best_move(matrix, player)

        # Build and send move string
        move_string = build_move_string(best_move)

        print(move_string)

        # Uncomment the line below if you want to send the move
        send_move(serial_connection, move_string)

    elif player == 2:  # human
        wait_for_command(serial_connection, "/end")

    # Uncomment the line below if you want to make a move for the human player
    # matrix = make_move(matrix, best_move)

    # Update the board with the new matrix
    redraw_board(board, matrix)

    if player == 1:
        score1 += 1
    elif player == 2:
        score2 += 1

# Function to start the game when Enter key is pressed
def start_game(event):
    global matrix
    matrix = [
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

# Close the serial connection when done
# serial_connection.close()
