import serial
import serial.tools.list_ports
    
def display_ports():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        print(port)


class Balance(serial.Serial):

    def __init__(self,
                 port='COM4', baudrate=19200):
        super().__init__()
        self.port=port
        self.baudrate = baudrate
        #self.parity = serial.PARITY_EVEN
        self.open()
        self.write(b'T\n')
        print(self.readline())

    def zero_balance(self):
        pass
    
    def get_mass(self):
        pass
        


if __name__ == '__main__':
    
  
    b=Balance()