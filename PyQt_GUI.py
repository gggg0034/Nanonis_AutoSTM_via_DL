import sys
import threading
from multiprocessing import Process, Queue

from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QMainWindow,
                             QPushButton, QVBoxLayout, QWidget)




def instruct_queue(q):
    """
    Thread function to monitor a queue for special characters.
    :param q: The queue to monitor.
    :param identifier: A string to identify which queue (and hence which button) this thread is monitoring.
    """
    while True:
        item = q.get()  # Retrieve the next item from the queue.
        # Check for a special character and perform an action.
        if item == "Home":
            print(f"set Scan frame to center")
        if item == "skip to next":
            print(f"skip to next")

def Pulse_queue(q):
    """
    Thread function to monitor a queue for special characters.
    :param q: The queue to monitor.
    :param identifier: A string to identify which queue (and hence which button) this thread is monitoring.
    """
    while True:
        item = q.get()  # Retrieve the next item from the queue.
        # Check for a special character and perform an action.
        if item[0] == "Pulse":
            print(f"Pulse {item[1]}V")


def TipShaper_queue(q):
    """
    Thread function to monitor a queue for special characters.
    :param q: The queue to monitor.
    :param identifier: A string to identify which queue (and hence which button) this thread is monitoring.
    """
    while True:
        item = q.get()  # Retrieve the next item from the queue.
        # Check for a special character and perform an action.
        if item[0] == "TipShaper":
            print(f"TipShaper {item[1]}")



def create_pyqt_app(q1, q2, q3):
    """
    Function to create and run the PyQt application with many buttons.
    :param q1: The first queue for communication.
    :param q2: The second queue for communication.
    """
    app = QApplication(sys.argv)  # Initialize the PyQt application.

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.voltage_list = [-3, -4, -6]
            self.height_list = ['-2n', '-4n', '-6n']

            self.setWindowTitle('STM Control Window')
            self.setGeometry(1000, 600, 600, 400)
            self.initUI()


        def initUI(self):
            # Create the central widget
            centralWidget = QWidget(self)
            self.setCentralWidget(centralWidget)

            mainLayout = QHBoxLayout()  # Create a vertical layout.
            # Create the left side layout (vertical)
            leftLayout = QVBoxLayout()
            self.button1 = QPushButton('Home')  # Create the first button.
            self.button1.clicked.connect(lambda: q1.put('Home'))  # Connect button to put special char in q1.
            leftLayout.addWidget(self.button1)  # Add the first button to the layout.
            
            self.button2 = QPushButton('skip to next')  # Create the second button.
            self.button2.clicked.connect(lambda: q1.put('skip to next'))  # Connect button to put special char in q2.
            leftLayout.addWidget(self.button2)  # Add the second button to the layout.
            
            midLayout  = QVBoxLayout() # Create a vertical layout.
            # Create the right side layout (vertical)
            
            self.Pulsebutton = QPushButton(f'Pulse {self.voltage_list[0]}V')  # Create the third button.
            self.Pulsebutton.clicked.connect(lambda: q2.put(['Pulse',self.voltage_list[0]]))  # Connect button to put ['Pulse',-3] q2.
            midLayout .addWidget(self.Pulsebutton) # Add the third button to the layout.

            self.Pulsebutton = QPushButton(f'Pulse {self.voltage_list[1]}V') 
            self.Pulsebutton.clicked.connect(lambda: q2.put(['Pulse',self.voltage_list[1]])) 
            midLayout .addWidget(self.Pulsebutton)

            self.Pulsebutton = QPushButton(f'Pulse {self.voltage_list[2]}V')
            self.Pulsebutton.clicked.connect(lambda: q2.put(['Pulse',self.voltage_list[2]])) 
            midLayout .addWidget(self.Pulsebutton) 

            rightLayout  = QVBoxLayout() # Create a vertical layout.
        
            self.TipShaperbutton = QPushButton(f'TipShaper {self.height_list[0]}')
            self.TipShaperbutton.clicked.connect(lambda: q3.put(['TipShaper', self.height_list[0]]))  # Connect button to put ['TipShaper', '-2n'] in q3.
            rightLayout .addWidget(self.TipShaperbutton) 

            self.TipShaperbutton = QPushButton(f'TipShaper {self.height_list[1]}')
            self.TipShaperbutton.clicked.connect(lambda: q3.put(['TipShaper', self.height_list[1]]))
            rightLayout .addWidget(self.TipShaperbutton) 

            self.TipShaperbutton = QPushButton(f'TipShaper {self.height_list[2]}')
            self.TipShaperbutton.clicked.connect(lambda: q3.put(['TipShaper', self.height_list[2]]))
            rightLayout .addWidget(self.TipShaperbutton) 


            # Add the left and right layouts to the main layout
            mainLayout.addLayout(leftLayout)
            mainLayout.addLayout(midLayout)
            mainLayout.addLayout(rightLayout)
            centralWidget.setLayout(mainLayout)  # Set the container as the central widget of the window.

    window = MainWindow()  # Instantiate the main window.
    window.show()  # Display the main window.
    sys.exit(app.exec_())  # Start the PyQt application's event loop.

if __name__ == "__main__":
    queue1 = Queue()  # Create the first queue.
    queue2 = Queue()  # Create the second queue.
    queue3 = Queue()  # Create the third queue.
       
    # Start the PyQt application in a child process with two queues.
    pyqt_process = Process(target=create_pyqt_app, args=(queue1, queue2, queue3,))
    pyqt_process.start()

    # Start two threads to monitor each queue.
    monitoring_thread1 = threading.Thread(target=instruct_queue, args=(queue1,))
    monitoring_thread2 = threading.Thread(target=Pulse_queue, args=(queue2,))
    monitoring_thread3 = threading.Thread(target=TipShaper_queue, args=(queue3,))
    
    monitoring_thread1.start()
    monitoring_thread2.start()
    monitoring_thread3.start()
    
    # Wait for the PyQt process to finish if necessary.
    pyqt_process.join()
