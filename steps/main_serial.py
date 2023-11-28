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
    ser = serial.Serial(port, 9600)
    ser.timeout = 2
    return ser

def notify_microcontroller(serial_connection):
    serial_connection.write(b'R')
    time.sleep(0.1)

def build_move_string(move, remove):
    string = 'move/'
    
    for index in range(len(move)):
        x, y = move[index]
        string += (str(x) + '-' + str(y) + '/')

    if remove is not None:
        string += 'remove/'
        for index in range(len(remove)):
            x, y = remove[index]
            string += (str(x) + '-' + str(y) + '/')

    return string

def send_move(serial_connection, move):
    serial_connection.write(move)

# Choose the COM port dynamically
#chosen_port = choose_serial_port()

# Initialize the serial connection
#serial_connection = initialize_serial_connection(chosen_port)

# Example move
move = ((4, 5), (6, 7))
remove = [(5, 6)]

# Notify microcontroller
#notify_microcontroller(serial_connection)

# Build and send move string
move_string = build_move_string(move, remove)
print(move_string)
#send_move(serial_connection, move_string)

# Close the serial connection when done

#serial_connection.close()
