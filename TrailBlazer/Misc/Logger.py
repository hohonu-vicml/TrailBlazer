import functools
import logging

from io import StringIO  # Python3

import sys

class SilencedStdOut:
    # https://stackoverflow.com/questions/65608502/is-there-a-way-to-force-any-function-to-not-be-verbose-in-python
    def __enter__(self):
        self.old_stdout = sys.stdout
        self.result = StringIO()
        sys.stdout = self.result

    def __exit__(self, *args, **kwargs):

        sys.stdout = self.old_stdout
        result_string = self.result.getvalue() # use if you want or discard.

class CustomFormatter(logging.Formatter):

    GRAY = "\x1b[38m"
    YELLOW = "\x1b[33m"
    CYAN = "\x1b[36m"
    RED = "\x1b[31m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"
    FORMAT = "[%(asctime)s - %(name)s - %(levelname)8s] - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: GRAY + FORMAT + RESET,
        logging.INFO: GRAY + FORMAT + RESET,
        logging.WARNING: YELLOW + FORMAT + RESET,
        logging.ERROR: RED + FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + FORMAT + RESET,
        logging.DEBUG: CYAN + FORMAT + RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# create logger with 'spam_application'

logger = logging.getLogger("TrailBlazer")
logger.handlers = []
logger.setLevel(logging.DEBUG)
# create console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

console_handler.setFormatter(CustomFormatter())
logger.addHandler(console_handler)

critical = logger.critical
fatal = logger.fatal
error = logger.error
warning = logger.warning
warn = logger.warn
info = logger.info
debug = logger.debug

if __name__ == "__main__":
    from DirectedDiffusion import Logger as log
    log.info("info message")
    log.warning("warning message")
    log.error("error message")
    log.debug("debug message")
    log.critical("critical message")
