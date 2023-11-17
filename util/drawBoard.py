# checkers_board_functions.py
import tkinter as tk

# Global color variables
BACKGROUND_COLOR = "#111111"
DARK_SQUARE_COLOR = "#8b4513"
LIGHT_SQUARE_COLOR = "#deb887"
CHECKER_PLAYER1_COLOR = "#202020"
CHECKER_PLAYER2_COLOR = "#f5f5f5"
CHECKER_OUTLINE_COLOR = "black"

def create_board(root, board_margin=20):
    margin_frame = tk.Frame(root, bg=BACKGROUND_COLOR, padx=board_margin, pady=board_margin)
    margin_frame.pack(fill=tk.BOTH, expand=True)

    board = tk.Canvas(margin_frame, bg=BACKGROUND_COLOR, highlightthickness=0)
    board.pack(fill=tk.BOTH, expand=True)

    root.bind("<Configure>", lambda event: redraw_board(board))

    redraw_board(board)
    return board

def redraw_board(board, checker_matrix=None):
    board.delete("all")
    board_width = board.winfo_width()
    board_height = board.winfo_height()

    draw_checkers(board, checker_matrix, board_width, board_height)


def draw_checkers(board, checker_matrix, board_width, board_height):
    square_size = min(board_width, board_height) / 8

    for row in range(8):
        for col in range(8):
            color = DARK_SQUARE_COLOR if (row + col) % 2 == 1 else LIGHT_SQUARE_COLOR
            x1, y1 = col * square_size, row * square_size
            x2, y2 = x1 + square_size, y1 + square_size
            board.create_rectangle(x1, y1, x2, y2, fill=color, outline=CHECKER_OUTLINE_COLOR)

            if checker_matrix is not None:
                checker = checker_matrix[row][col]
                if checker != 0:
                    place_checker(board, row, col, square_size, checker)

def place_checker(board, row, col, square_size, checker_type):
    checker_width = 0.8 * square_size
    x = col * square_size + (square_size - checker_width) / 2
    y = row * square_size + (square_size - checker_width) / 2

    color = None

    if checker_type == 1:
        color = CHECKER_PLAYER1_COLOR
    elif checker_type == 2:
        color = CHECKER_PLAYER2_COLOR
    elif checker_type == 3:
        color = CHECKER_PLAYER1_COLOR
    elif checker_type == 4:
        color = CHECKER_PLAYER2_COLOR

    board.create_oval(x, y, x + checker_width, y + checker_width, fill=color, outline=CHECKER_OUTLINE_COLOR)

    if checker_type - 2 > 0:
        draw_crown(board, x, y, checker_width)

def draw_crown(board, x, y, checker_width):
    crown_width = 0.6 * checker_width
    crown_height = 0.3 * checker_width

    x_crown = x + checker_width / 2
    y_crown = y + checker_width / 2

    board.create_polygon(
        x_crown - crown_width / 2, y_crown + crown_height / 2,
        x_crown - 0.4 * crown_width, y_crown + crown_height / 2 - crown_height,
        x_crown - 0.2 * crown_width, y_crown + crown_height / 2 - crown_height,
        x_crown, y_crown + crown_height / 2 - 1.5 * crown_height,
        x_crown + 0.2 * crown_width, y_crown + crown_height / 2 - crown_height,
        x_crown + 0.4 * crown_width, y_crown + crown_height / 2 - crown_height,
        x_crown + crown_width / 2, y_crown + crown_height / 2,
        fill="gold", outline="black"
    )

def run_mainloop(root):
    root.mainloop()


