from utils.log import *

def test():
    INFO("This is a test message")
    DEBUG("This is a test message")
    WARNING("This is a test message")
    ERROR("This is a test message")
    CRITICAL("This is a test message")

if __name__ == "__main__":
    test()