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

def build_move_string(moveTupple):
    move = []
    remove = []
    for index in range(len(moveTupple)):
        
    string = ''
    
    for index in range(len(move)):
        x, y = move[index]
        string += (str(x) + '-' + str(y))
        if index != len(move)-1:
            string += '/'

    if remove is not None:
        string += '*'
        for index in range(len(remove)):
            x, y = remove[index]
            string += (str(x) + '-' + str(y))
            if index != len(remove)-1:
                string += '/'

    return string

def send_move(serial_connection, move):
    serial_connection.write(move)

def wait_for_start(serial_connection):
    while True:
        # Wait for data from the serial port
        if serial_connection.in_waiting > 0:
            received_data = serial_connection.readline().decode().strip()

            # Check if the received data is the "start" signal
            if received_data == "start":
                print("Received 'start' signal from the microcontroller.")
                break
            else:
                print(f"Unexpected data received: {received_data}")

        # Add a small delay to avoid high CPU usage in the loop
        time.sleep(0.1)

