import os

ROOT_PATH = "/Users/dong/GitHub/learning-dtmc/"

STM = "state_transition_matrix"
STWM = "state_transition_weight_matrix"
PLABEL = "prism_label"
FFREQ = "id2final_freq"
ID2FREQ = "id2frequency"
ID2CHILDREN = "id2children"
ID2PARENT = "id2parent"
ID2ACTIONS = "id2actions"
IDMERGED = "merged_prefix_id"

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