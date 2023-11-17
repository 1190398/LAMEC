# checkers_board.py
import tkinter as tk

class CheckersBoard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Checkers Board")
        self.fullscreen = False
        self.default_width = 800
        self.default_height = 600
        self.geometry("800x600")
        self.configure(bg="#202020")
        self.create_board()
        self.bind("<Configure>", self.on_window_resize)
        self.bind("<F11>", self.toggle_fullscreen)
        self.bind("<Escape>", self.exit_fullscreen)
        self.checker_matrix = None

    def create_board(self):
        self.board_margin = 20
        self.update_board_size()

        self.board = tk.Canvas(self, width=self.board_width, height=self.board_height, bg="#111111")  # Dark gray background
        self.board.place(x=self.board_margin, y=self.board_margin)

        for row in range(8):
            for col in range(8):
                color = "#8b4513" if (row + col) % 2 == 1 else "#deb887"
                square_size = self.board_width / 8
                x1, y1 = col * square_size, row * square_size
                x2, y2 = x1 + square_size, y1 + square_size
                self.board.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

    def update_board_size(self):
        window_width = self.winfo_width()
        window_height = self.winfo_height()

        board_size = min(window_width, window_height) - 2 * self.board_margin
        self.board_width = board_size
        self.board_height = board_size

    def on_window_resize(self, event):
        self.update_board_size()
        self.board.config(width=self.board_width, height=self.board_height)
        self.board.place(x=self.board_margin, y=self.board_margin)
        self.redraw_board()

    def toggle_fullscreen(self, event):
        self.fullscreen = not self.fullscreen
        self.attributes('-fullscreen', self.fullscreen)
        if not self.fullscreen:
            self.geometry(f"{self.default_width}x{self.default_height}")

    def exit_fullscreen(self, event):
        self.fullscreen = False
        self.attributes('-fullscreen', False)
        self.geometry(f"{self.default_width}x{self.default_height}")

    def redraw_board(self):
        self.board.delete("all")
        self.draw_checkers()

    def draw_checkers(self, checker_matrix=None):
        square_size = self.board_width / 8
        if checker_matrix is None:
            checker_matrix = get_default_matrix()

        for row in range(8):
            for col in range(8):
                color = "#8b4513" if (row + col) % 2 == 1 else "#deb887"
                x1, y1 = col * square_size, row * square_size
                x2, y2 = x1 + square_size, y1 + square_size
                self.board.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

                checker = checker_matrix[row][col]
                if checker == 1:
                    self.place_checker(row, col, color="#202020")
                elif checker == 2:
                    self.place_checker(row, col, color="#f5f5f5")

    def place_checker(self, row, col, color):
        square_size = self.board_width / 8
        checker_width = 0.8 * square_size
        x = col * square_size + (square_size - checker_width) / 2
        y = row * square_size + (square_size - checker_width) / 2
        self.board.create_oval(x, y, x + checker_width, y + checker_width, fill=color, outline="black")

    def update_checkers(self, checker_matrix):
        self.checker_matrix = checker_matrix
        self.redraw_board()

    def run(self):
        self.mainloop()


def get_default_matrix():
    return [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
