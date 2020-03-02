from aalergia.aalergia import AALERGIA
from utils.read_seq import *
from utils.constant import *
from utils.time_util import *

def test_aalergia():

    sequence, alphabet = load_data(get_path(SeqPath.HERMAN7), symbols_count=100)
    print(current_timestamp(), "init")
    al = AALERGIA(64, sequence, alphabet, start_symbol="0000000", output_path="./")
    print(current_timestamp(), "learing....")
    dffa = al.learn()
    print(current_timestamp(), "done")
    al.output_prism(dffa, "herman7_")


if __name__ == '__main__':
    test_aalergia()
