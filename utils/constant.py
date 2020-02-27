import os

ROOT_PATH = "/Users/dong/GitHub/learning-dtmc/"

def get_path(p):
        return os.path.join(ROOT_PATH,p)

class SeqPath:
        CRAPS = 'data/craps/data.txt'
        DICE = 'data/dice/data.txt'
        HERMAN3 = 'data/herman3/data.txt'
        HERMAN7 = 'data/herman7/data.txt'
        HERMAN11 = 'data/herman11/data.txt'
        HERMAN19 = 'data/herman19/data.txt'
        HERMAN21 = 'data/herman21/data.txt'