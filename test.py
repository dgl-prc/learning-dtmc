from alergia import *
from aalergia.aalergia import AALERGIA
from utils.read_seq import *
from utils.constant import *
from utils.time_util import *


# def test_print_pdfa():
#     trans_func = {1: {'a': 2, 'b': 3}, 2: {'a': 1, 'b': 2}}
#     trans_w_func = {1: {'a': 0.2, 'b': 0.8}, 2: {'a': 0.4, 'b': 0.6}}
#     alphabet = ['a', 'b']
#     alphabet.sort()
#     print_pdfa(trans_func, trans_w_func, alphabet)


def test_alergia1():
    sequence = [['', "1", "1", "0"], [''], [''], [''], ['', '0'], [''], ['', '0', '0'], ['', '0', '0'], [''], [''],
                [''],
                ['', '1', '0', '1', '1', '0'], [''], [''], ['', '1', '0', '0']]
    alphabet = ["1", "0"]
    al = ALERGIA(0.8, sequence, alphabet, start_symbol='', output_path="./", t_max_length=10000)
    al.learn()


#
def test_alergia2():
    sequence, alphabet = load_data(get_path(SeqPath.CRAPS))
    al = ALERGIA(0.008, sequence, alphabet, start_symbol="start", output_path="./", t_max_length=1000)
    al.learn()


def test_aalergia():
    # for symbol_count in range(50, 105, 5):
    #     sequence, alphabet = load_data(get_path(SeqPath.CRAPS), symbols_count=symbol_count)
    #     al = AALERGIA(64, sequence, alphabet, start_symbol="start", output_path="./")
    #     dffa = al.learn()
    #     al.pretty_look(dffa[STM], dffa[STM])
    #     al.output_prism(dffa)

    sequence, alphabet = load_data(get_path(SeqPath.HERMAN19), symbols_count=50)
    print(current_timestamp(), "init")
    al = AALERGIA(64, sequence, alphabet, start_symbol="stable19", output_path="./")
    print(current_timestamp(), "learing....")
    dffa = al.learn()
    al.pretty_look(dffa[STM], dffa[STM])
    print(current_timestamp(), "done")
    al.output_prism(dffa, "herman3_")


if __name__ == '__main__':
    test_aalergia()
    # test_alergia2()
    # test_alergia1()
    # test_print_pdfa()
