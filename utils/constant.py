import os

ROOT_PATH = "/Users/dong/GitHub/learning-dtmc/"

def get_path(p):
        return os.path.join(ROOT_PATH,p)

class SeqPath:
        CRAPS = 'data/craps/seq.txt'