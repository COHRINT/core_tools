#!/usr/bin/env python

'''Interface for entering observations for Cops and Robots POMDP experiments.
'''

__author__ = "Ian Loefgren"
__copyright__ = "Copyright 2017, Cohrint"
__credits__ = ["Ian Loefgren"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Ian Loefgren"
__email__ = "ian.loefgren@colorado.edu"
__status__ = "Development"

import sys
import rospy
import matplotlib
import time

matplotlib.use('Qt5Agg')
import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from rqt_gui.main import Main

from observation_interface.srv import *
from observation_interface.msg import *

class ObsInterfaceWidget(QWidget):
    obs_ready = pyqtSignal()
    poll_signal = pyqtSignal()

    def __init__(self):
        super(QWidget,self).__init__()
        self.initUI()

        rospy.init_node('obs_interface')
        self.s = rospy.Service('observation_interface',observation_server,self.obs_handler)

        self.obs_ready.connect(self.switch_obs_state)
        self.obs_ready.connect(self.clear_obs_prompt)
        self.poll_signal.connect(self.set_obs_prompt)
        self.obs_state = False
        self.obs = None

        print('Observation Interface ready.')

    def initUI(self):
        self.main_widget = QWidget(self)
        self.main_layout = QHBoxLayout(self.main_widget)

        self.pltCanvas = MplCanvas(self.main_widget, width=5, height=4, dpi=100 )

        self.vert_layout = QVBoxLayout()
        self.vert_layout_obs = QVBoxLayout()
        self.vert_layout_func = QVBoxLayout()
        self.horiz_layout_btns = QHBoxLayout()
        self.grid_layout = QGridLayout()

        self.app_title = QLabel('User Input')
        font = QFont()
        font.setPointSize(20)
        self.app_title.setFont(font)
        self.app_title.setAlignment(Qt.AlignHCenter)
        self.vert_layout.addWidget(self.app_title)

        self.obs_prompt = QLabel()
        self.obs_prompt.setStyleSheet("QLabel { color : red; }")
        font.setPointSize(10)
        self.obs_prompt.setFont(font)
        self.obs_prompt.setAlignment(Qt.AlignHCenter)
        self.vert_layout.addWidget(self.obs_prompt)

        self.up_btn = QPushButton('UP',self)
        self.up_btn.clicked.connect(self.up_input)
        self.vert_layout_obs.addWidget(self.up_btn)

        self.down_btn = QPushButton('DOWN',self)
        self.down_btn.clicked.connect(self.down_input)
        self.vert_layout_obs.addWidget(self.down_btn)

        self.left_btn = QPushButton('LEFT',self)
        self.left_btn.clicked.connect(self.left_input)
        self.vert_layout_obs.addWidget(self.left_btn)

        self.right_btn = QPushButton('RIGHT',self)
        self.right_btn.clicked.connect(self.right_input)
        self.vert_layout_obs.addWidget(self.right_btn)

        self.near_btn = QPushButton('NEAR',self)
        self.near_btn.clicked.connect(self.near_input)
        self.vert_layout_obs.addWidget(self.near_btn)

        self.vert_layout_obs.addWidget(QHLine())

        self.send_btn = QPushButton('SEND OBSERVATION',self)
        self.send_btn.clicked.connect(self.send_obs)
        self.vert_layout_obs.addWidget(self.send_btn)

        self.vert_layout_obs.addWidget(QHLine())

        self.quit_btn = QPushButton('QUIT',self)
        self.quit_btn.clicked.connect(self.close)
        self.vert_layout_obs.addWidget(self.quit_btn)

        self.vert_layout.addLayout(self.vert_layout_obs)

        self.main_layout.addLayout(self.vert_layout)
        self.main_layout.addWidget(self.pltCanvas)
        self.setLayout(self.main_layout)

        self.setGeometry(100,0,250,150)
        self.setFixedSize(950,550)
        self.setWindowTitle('User Input')
        self.show()

    def up_input(self):
        self.obs = 'UP'
        self.obs_num = 8
    def down_input(self):
        self.obs = 'DOWN'
        self.obs_num = 2
    def left_input(self):
        self.obs = 'LEFT'
        self.obs_num = 4
    def right_input(self):
        self.obs = 'RIGHT'
        self.obs_num = 6
    def near_input(self):
        self.obs = 'NEAR'
        self.obs_num = 5
    def send_obs(self):
        self.obs_ready.emit()

    @pyqtSlot()
    def switch_obs_state(self):
        self.obs_state = not self.obs_state

    @pyqtSlot()
    def set_obs_prompt(self):
        text = 'Please make an observation.\n Last observation was {}'.format(self.obs)
        self.obs_prompt.setText(text)

    @pyqtSlot()
    def clear_obs_prompt(self):
        self.obs_prompt.setText('')

    def obs_handler(self,req):
        '''
        wait for signal from send observation button
        '''
        self.poll_signal.emit()
        while not self.obs_state:
            PyQt5.QtCore.QCoreApplication.processEvents()
        self.switch_obs_state()
        msg = self.create_message(self.obs_num)
        return msg

    def create_message(self,obs):
        msg = ObservationResponse()
        msg.observation = obs
        return msg

class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        print 'Init of MplCanvas'
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.axes = fig.add_subplot(111)

        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Fixed,
                                   QSizePolicy.Fixed)
        FigureCanvas.updateGeometry(self)

    def update_canvas(self,belief):
        belief.plotSliceFrom4D()

class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


if __name__ == "__main__":
    # main = Main()
    # sys.exit(main.main(sys.argv,standalone='src.observation_interface.ObsInterfacePlugin'))

    app = QApplication(sys.argv)
    obs_app = ObsInterfaceWidget()
    sys.exit(app.exec_())
