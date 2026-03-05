import os
import sys
from PyQt6.QtWidgets import QApplication
import netx_gui



def main():
    app = QApplication(sys.argv)
    window = netx_gui.GUI()
    window.show()
    sys.exit(app.exec())



if __name__ == "__main__":
    main()
















