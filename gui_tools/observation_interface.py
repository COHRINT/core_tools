#!/usr/env/bin python

'''Interface for entering observations for Cops and Robots POMDP experiments.
'''

import sys
import subprocess
from subprocess import Popen, PIPE

from PyQt5.QtWidgets import *
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QFont

class UserInputWindow(QWidget):

    def __init__(self):
        super(QWidget,self).__init__()
        self.initUI()

    def initUI(self):

        self.horiz_layout = QHBoxLayout()
        self.vert_layout = QVBoxLayout()

        self.app_title = QLabel('User Input')
        font = QFont()
        font.setPointSize(20)
        self.app_title.setFont(font)
        self.app_title.setAlignment(Qt.AlignHCenter)
        self.vert_layout.addWidget(self.app_title)

        self.up_btn = QPushButton('UP',self)
        self.up_btn.clicked.connect(self.up_input)
        self.horiz_layout.addWidget(self.up_btn)

        self.down_btn = QPushButton('DOWN',self)
        self.down_btn.clicked.connect(self.down_input)
        self.horiz_layout.addWidget(self.down_btn)

        self.left_btn = QPushButton('LEFT',self)
        self.left_btn.clicked.connect(self.left_input)
        self.horiz_layout.addWidget(self.left_btn)

        self.right_btn = QPushButton('RIGHT',self)
        self.right_btn.clicked.connect(self.right_input)
        self.horiz_layout.addWidget(self.right_btn)

        self.vert_layout.addLayout(self.horiz_layout)

        self.setLayout(self.vert_layout)


        self.setGeometry(100,0,800,700)
        self.setWindowTitle('User Input')
        self.show()

    def up_input(self):
        self.send_input('up')

    def down_input(self):
        self.send_input('down')

    def left_input(self):
        self.send_input('left')

    def right_input(self):
        self.send_input('right')

    def send_input(self,input_):
        proc = subprocess.Popen('echo',stdin=PIPE,stdout=PIPE,stderr=PIPE,shell=True)
        output = ''
        if input_ == 'up':
            output = '8'
        elif input_ == 'down':
            output = '2'
        elif input_ == 'left':
            output = '4'
        elif input_ == 'right':
            output = '6'
        print(output)
        proc.communicate(input=output)[0]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    input_app = UserInputWindow()
    sys.exit(app.exec_())
