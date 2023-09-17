import logging


def get_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(filename)s:%(lineno)d]: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
