import sys
import logging
import threading
from .rich_utils import custom_console, MyRichHandler, MyReprHighlighter


def init_logger():
    # Logger
    # Change the logging stream from stderr to stdout to be compatible with horovod.
    patch_logging_stream(logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = MyRichHandler(highlighter=MyReprHighlighter(), console=custom_console)
    handler.setLevel(logging.INFO)
    formatter = DispatchingFormatter(
        {"fit_progressbar": logging.Formatter("%(message)s", datefmt=" [%X]")},
        logging.Formatter("%(message)s", datefmt="[%X]"),
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# Reference: https://stackoverflow.com/questions/1383254/logging-streamhandler-and-standard-streams/55494220#55494220
def _logging_handle(self, record):
    self.STREAM_LOCKER = getattr(self, "STREAM_LOCKER", threading.RLock())
    if self.stream in (sys.stdout, sys.stderr) and record.levelname in self.FIX_LEVELS:
        try:
            self.STREAM_LOCKER.acquire()
            self.stream = sys.stdout
            self.old_handle(record)
            self.stream = sys.stderr
        finally:
            self.STREAM_LOCKER.release()
    else:
        self.old_handle(record)


def patch_logging_stream(*levels):
    """
    writing some logging level message to sys.stdout

    example:
    patch_logging_stream(logging.INFO, logging.DEBUG)
    logging.getLogger('root').setLevel(logging.DEBUG)

    logging.getLogger('root').debug('test stdout')
    logging.getLogger('root').error('test stderr')
    """
    stream_handler = logging.StreamHandler
    levels = levels or [logging.DEBUG, logging.INFO]
    stream_handler.FIX_LEVELS = [logging.getLevelName(i) for i in levels]
    if hasattr(stream_handler, "old_handle"):
        stream_handler.handle = stream_handler.old_handle
    stream_handler.old_handle = stream_handler.handle
    stream_handler.handle = _logging_handle


class ColoredFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt=None, datefmt=None, style='%', validate=True):
        self.FORMATS = {
            logging.DEBUG: self.grey + fmt + self.reset,
            logging.INFO: self.grey + fmt + self.reset,
            logging.WARNING: self.yellow + fmt + self.reset,
            logging.ERROR: self.red + fmt + self.reset,
            logging.CRITICAL: self.bold_red + fmt + self.reset
        }
        self._datefmt = datefmt
        self._style = style
        self._validate = validate

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, self._datefmt, self._style, self._validate)
        return formatter.format(record)


class DispatchingFormatter:
    """
    Dispatch formatter for logger and it's sub logger.
    https://stackoverflow.com/questions/1741972/how-to-use-different-formatters-with-the-same-logging-handler-in-python
    """
    def __init__(self, formatters, default_formatter):
        self._formatters = formatters
        self._default_formatter = default_formatter

    def format(self, record):
        # Search from record's logger up to it's parents:
        logger = logging.getLogger(record.name)
        while logger:
            # Check if suitable formatter for current logger exists:
            if logger.name in self._formatters:
                formatter = self._formatters[logger.name]
                break
            else:
                logger = logger.parent
        else:
            # If no formatter found, just use default:
            formatter = self._default_formatter
        self.datefmt = formatter.datefmt
        return formatter.format(record)
