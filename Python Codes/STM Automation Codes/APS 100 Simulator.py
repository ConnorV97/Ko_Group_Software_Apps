import serial
# print(serial)
# print(serial.__file__)
import time

"""

This program is designed to simulate communication with the APS 100 by Unisoku. APS 100 utilizes a 
ASCII communication protocol and expects a carriage return delimiter at the end of each message received.

To test this on a stand alone computer it is recommend to use com0com for com port to com port communication. 
"""

# Set up port
port = serial.Serial('COM3', 9600, timeout=5)


# Initi voltage and current values
voltage = 0
current =0



while True:

    # Read port unitl CR
    cmd = port.read_until(b'\r').decode('utf-8').strip()


    # Set-up key commands to be simulated

    if cmd =="*IDN?":
        port.write(b"Unisoku,APS-100,SIM,1.0\r")
        print("Message received")

    elif cmd.startswith("VSET"):
        voltage = float(cmd.split()[1])
        port.write(b'Ok\r')


