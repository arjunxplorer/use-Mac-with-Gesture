"""Entry point: python -m gesturemac"""

import os

# Quieter MediaPipe / TFLite logs in the terminal.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from gesturemac.app import GestureMacApp


def main() -> None:
    GestureMacApp().run()


if __name__ == "__main__":
    main()
