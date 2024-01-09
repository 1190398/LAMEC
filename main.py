import tkinter as tk
from steps.brain import get_best_move, print_board
from steps.serial import choose_serial_port, initialize_serial_connection, wait_for_command, build_move_string, send_move
from steps.recon import main_recon, setup_web_cam
import time

# Choose the COM port dynamically
#chosen_port = choose_serial_port()

chosen_port = 3

# Initialize the serial connection
serial_connection = initialize_serial_connection(chosen_port)

#setup_web_cam()

player = 1
score1 = 0
score2 = 0

while True:

    # Wait for the "start" signal
    wait_for_command(serial_connection, "/start")

    matrix = main_recon()

    time.sleep(0.5)
    
    # Get the best move
    best_move, isQueen = get_best_move(matrix, player)

    time.sleep(0.5)

    # Build and send move string
    move_string = build_move_string(best_move, isQueen)

    time.sleep(0.5)

    # Uncomment the line below if you want to send the move
    send_move(serial_connection, move_string)

    time.sleep(0.5)

    wait_for_command(serial_connection, "/end")

# Close the serial connection when done
serial_connection.close()
