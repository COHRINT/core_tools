
from observation_interface_GUI import ObsInterfaceWidget
from PyQt5.QtGui.plugin import Plugin

class ObsInterfacePlugin(Plugin):
    def __init__(self,context):
        super(ObsInterfacePlugin,self).__init__(context)
        self._widget = ObsInterfaceWidget()

        context.add_widget(self._widget)

        self.setObjectName('Human Observation UI')
