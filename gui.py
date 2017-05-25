#!/usr/env/bin python

'''
Dynamically created user interface to select parameters for experiments run using
the Core Tools experiment testbed. All parameter options and their descriptions
are contained in the enum.yaml file also found in this directory.

To add parameters to the interface, they must be added to enum.yaml and follow
the structure that is decribed in that file.

The interface launches a shell script to then run the experiment. This shell
script can be changed in the run_experiment() function below.
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
import yaml

from PyQt5.QtWidgets import *
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QFont

class ParameterWindow(QWidget):

    def __init__(self):
        super(QWidget,self).__init__()
        self.initUI()

    def initUI(self):

        self.params = {'main':{},'robots_to_use':{},'robots':{}}

        self.horiz_layout = QHBoxLayout()
        self.vert_layout = QVBoxLayout()

        self.app_title = QLabel('Experiment Parameters')
        font = QFont()
        font.setPointSize(20)
        self.app_title.setFont(font)
        self.app_title.setAlignment(Qt.AlignHCenter)
        self.vert_layout.addWidget(self.app_title)

        # Make seperate tabs for main and robot parameters
        self.tabs = QTabWidget()
        self.main_tab = QWidget()
        self.robot_tab = QWidget()

        self.tabs.addTab(self.main_tab,'Main')
        self.tabs.addTab(self.robot_tab,'Robots')

        self.robot_tab.layout = QVBoxLayout()
        self.main_tab.layout = QVBoxLayout()
        self.make_menus()
        self.vert_layout.addWidget(self.tabs)

        # Create and connect load button to loading config
        self.load_btn = QPushButton('LOAD',self)
        self.load_btn.clicked.connect(self.load_config)
        self.horiz_layout.addWidget(self.load_btn)

        # Create and connect save button to saving config
        self.save_btn = QPushButton('SAVE',self)
        self.save_btn.clicked.connect(self.save_config)
        self.horiz_layout.addWidget(self.save_btn)

        # Create and connect run button to running experiment shell script
        self.run_btn = QPushButton('RUN',self)
        self.run_btn.clicked.connect(self.run_experiment)
        self.horiz_layout.addWidget(self.run_btn)

        # Create and connect quit button
        self.quit_btn = QPushButton('QUIT',self)
        self.quit_btn.clicked.connect(self.close)
        self.vert_layout.addWidget(self.quit_btn)

        self.vert_layout.addLayout(self.horiz_layout)

        self.setLayout(self.vert_layout)


        self.setGeometry(100,0,800,700)
        self.setWindowTitle('Core Tools Testbed')
        self.show()

    def show_file_dialog(self,open_type):
        '''Open the load file dialog menu. Returns false if dialog is cancelled.
        '''
        file_name = ''
        if open_type == 'load':
            file_name = QFileDialog.getOpenFileName(self,'Load config file', './')
        elif open_type == 'save':
            file_name = QFileDialog.getSaveFileName(self,'Save config file', './')

        if file_name:
            return file_name
        else:
            return False

    def generate_config(self):
        '''Parses current menu selctions using check_menus() and combines the
        main and robot paramter variables into one config dictionary.
        '''
        self.check_menus()
        self.cfg = {'main':self.params['main'],'robots':self.robot_states}

    def load_config(self):
        '''Calls load file dialog, loads selected config from yaml file into
        dictionary, and divides the main and robot parameter sub-dictionaries
        into their respective varaibles.
        '''
        file_name = self.show_file_dialog('load')

        cfg = None
        if file_name[0]:
            try:
                with open(file_name[0],'r') as stream:
                    cfg = yaml.load(stream)
            except:
                print('File not found or could not be opened!')
        else:
            return False

        self.params['main'] = cfg['main']
        self.robot_states = cfg['robots']

        self.update_menus()

    def save_config(self):
        '''Calls generate_config() to create config dictionary and dumps it to
        selected yaml file. safe_dump removes python encoding.
        '''
        self.generate_config()

        file_name = self.show_file_dialog('save')
        file_name = file_name[0]
        if not file_name:
            return
        with open(file_name,'w') as config_file:
            yaml.safe_dump(self.cfg,config_file,default_flow_style=False)

        print('Config file saved!')

    def run_experiment(self):
        '''Generates config to be used and creates new process to run experiment
        shell script. Robots to be used passed to shell script but all other params
        still read from saved config in main_tester.py.
        '''
        import subprocess
        self.generate_config()

        use_robots = [self.cfg['robots']['Deckard']['use'], \
                        self.cfg['robots']['Roy']['use'], \
                        self.cfg['robots']['Pris']['use'], \
                        self.cfg['robots']['Zhora']['use']]

        print(use_robots)

        p = subprocess.Popen(["./run.sh",str(int(use_robots[0])), \
                                        str(int(use_robots[1])), \
                                        str(int(use_robots[2])), \
                                        str(int(use_robots[3]))])
        p.wait()

    def make_buttons(self):
        pass

    def load_options(self):
        '''Loads enum file and creates empty dictionaries for each robot listed,
        as well as empty strings for each option for each robot.
        '''
        enum_file = 'enum.yaml'
        with open(enum_file,'r') as stream:
            self.options = yaml.load(stream)

        self.robot_states = {}
        for robot in self.options['robots']:
            self.robot_states[robot] = {}
            for option in self.options['robot_params']:
                self.robot_states[robot][option] = ''

    def check_menus(self):
        '''Iterate over all selections in main and robot parameter menus and store
        selections in the main and robot parameter dictionaries.
        '''
        for option in self.main_options:
            data_type = option[2]
            key_str = option[1]
            value = ''

            if data_type == 'bool':
                value = option[0].isChecked()
            elif data_type == 'int':
                value = int(option[0].text())
            elif data_type == 'str':
                value = option[0].currentText()

            self.params['main'][key_str] = value

        for checkbox in self.robot_checkboxes:
            robot = checkbox.text()
            use_value = ''
            self.robot_states[robot]
            if checkbox.isChecked():
                use_value = True
            else:
                use_value = False
            self.robot_states[robot]['use'] = use_value
            for param in self.robot_options[robot]:
                if param == 'goal_planner_cfg':
                    param_value = self.robot_options[robot][param].currentText()
                    self.robot_states[robot][param] = {}
                    self.robot_states[robot][param]['type_'] = param_value
                else:
                    param_value = self.robot_options[robot][param].currentText()
                    self.robot_states[robot][param] = param_value

    def update_menus(self):
        '''Update menu selections for the main and robot parameters to match the
        main and robot parameter variables.
        '''
        for option in self.main_options:
            data_type = option[2]
            value = self.params['main'][option[1]]
            if data_type == 'bool':
                option[0].setChecked(value)
            elif data_type == 'int':
                option[0].setText(str(value))
            elif data_type == 'str':
                index = option[0].findText(str(value))
                option[0].setCurrentIndex(index)

        for checkbox in self.robot_checkboxes:
            robot = checkbox.text()
            value = self.robot_states[robot]['use']
            checkbox.setChecked(value)

            for param in self.robot_options[robot]:
                if param == 'goal_planner_cfg':
                    param_dropdown = self.robot_options[robot][param]
                    param_value = self.robot_states[robot][param]['type_']
                    index = param_dropdown.findText(str(param_value))
                    param_dropdown.setCurrentIndex(index)
                else:
                    param_dropdown = self.robot_options[robot][param]
                    param_value = self.robot_states[robot][param]
                    index = param_dropdown.findText(str(param_value))
                    param_dropdown.setCurrentIndex(index)


    def make_menus(self):
        '''Create the main and robot parameter menus.
        '''
        self.load_options()
        self.make_main_menu()
        self.make_robot_menu()

    def make_main_menu(self):
        '''Create the menu for the main parameters by iterating over the options
        loaded from the enum file. Creates checkboxes for bool parameters, line
        edits for int parameters, and dropdowns for str parameters. Also adds the
        tool tips to each widget from the enum file.
        '''
        self.main_options = []

        for option in self.options['main']:
            data_type = self.options['main'][option]['data_type']
            tool_tip = self.options['main'][option]['tool_tip']
            option_select = ''
            option_title = ''

            if data_type == 'bool':
                option_select = QCheckBox(option)

            elif data_type == 'int':
                option_title = QLabel(option)
                option_select = QLineEdit(self.options['main'][option]['values'][0])
                self.main_tab.layout.addWidget(option_title)

            elif data_type == 'str':
                option_title = QLabel(option)
                option_select = QComboBox()
                for detail_option in self.options['main'][option]['values']:
                    option_select.addItem(detail_option)
                self.main_tab.layout.addWidget(option_title)

            self.main_options.append((option_select,option,data_type))

            option_select.setToolTip(tool_tip)
            self.main_tab.layout.addWidget(option_select)

        self.main_tab.setLayout(self.main_tab.layout)

    def make_robot_menu(self):
        '''Make the robot parameter menu by iterating over the options loaded
        from the enum file.
        '''
        self.robot_checkboxes = []
        self.robot_options = {}
        for robot in self.options['robots']:
            robot_checkbox = QCheckBox(robot)
            self.robot_checkboxes.append(robot_checkbox)
            self.robot_tab.layout.addWidget(robot_checkbox)
            self.robot_options[robot] = {}
            for option in self.options['robot_params']:
                option_title = QLabel(option)
                option_dropdown = QComboBox()
                self.robot_options[robot][option] = option_dropdown
                for detail_option in self.options['robot_params'][option]:
                    if detail_option == 'type_':
                        for type_ in self.options['robot_params'][option][detail_option]:
                            option_dropdown.addItem(type_)
                    else:
                        option_dropdown.addItem(detail_option)
                self.robot_tab.layout.addWidget(option_title)
                self.robot_tab.layout.addWidget(option_dropdown)

        self.robot_tab.setLayout(self.robot_tab.layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    coretools_app = ParameterWindow()
    sys.exit(app.exec_())
