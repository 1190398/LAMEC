import serial
import time
from serial.tools import list_ports

def choose_serial_port():
    # List available serial ports and let the user choose one
    available_ports = list_ports.comports()
    
    print("Available COM Ports:")
    for port, desc, hwid in sorted(available_ports):
        print(f"{port}: {desc} [{hwid}]")
    
    chosen_port = input("Enter the COM port you want to use: ")
    return chosen_port

def initialize_serial_connection(port):
    ser = serial.Serial('COM'+str(port), 9600)
    return ser

def build_move_string(moveTupple, isQueen):
    remove = []
    for index in range(len(moveTupple)-1):
        x1, y1 = moveTupple[index]
        x2, y2 = moveTupple[index+1]

        # Transform y-coordinates for endpoints
        #y1 = 7 - y1
        #y2 = 7 - y2

        middle_x = (x1 + x2) // 2
        middle_y = (y1 + y2) // 2

        # Check if the move is an eating move
        is_eating_move = abs(x1 - x2) == 2

        if is_eating_move:
            remove.append((middle_x, middle_y))

    string = ''

    for index in range(len(moveTupple)):
        x, y = moveTupple[index]
        # Transform y-coordinate for endpoints
        y = 7 - y
        string += (str(x) + '-' + str(y))
        if index != len(moveTupple)-1:
            string += '/'

    if len(remove) > 0:
        string += '*'
        for index in range(len(remove)):
            x, y = remove[index]
            # Transform y-coordinate for middle points
            y = 7 - y
            string += (str(x) + '-' + str(y))
            if index != len(remove)-1:
                string += '/'

    if isQueen:
        string += "*T"

    string += '\0'
    return string


def send_move(serial_connection, move):
    move_with_newline = move + '\n'
    serial_connection.write(move_with_newline.encode())
    #print(move_with_newline)


def wait_for_command(serial_connection, command):
    while True:
        # Wait for data from the serial port
        if serial_connection.in_waiting > 0:
            received_data = serial_connection.readline().decode().strip()

            if received_data == command:
                print(f"Command: {command}")
                break
            else:
                print(received_data)

        # Add a small delay to avoid high CPU usage in the loop
        time.sleep(0.1)

