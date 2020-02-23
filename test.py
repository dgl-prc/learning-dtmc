from alergia import ALERGIA
from aalergia.aalergia import AALERGIA
from visulize import visulize_tree, print_pdfa
from utils.read_seq import *


# def test_print_pdfa():
#     trans_func = {1: {'a': 2, 'b': 3}, 2: {'a': 1, 'b': 2}}
#     trans_w_func = {1: {'a': 0.2, 'b': 0.8}, 2: {'a': 0.4, 'b': 0.6}}
#     alphabet = ['a', 'b']
#     alphabet.sort()
#     print_pdfa(trans_func, trans_w_func, alphabet)


def test_alergia1():
    sequence = [['',"1", "1", "0"], [''], [''], [''], ['','0'], [''], ['','0', '0'], ['','0', '0'], [''], [''], [''],
                ['','1', '0', '1', '1', '0'], [''], [''], ['','1', '0', '0']]
    alphabet = ["1","0"]
    al = ALERGIA(0.8, sequence, alphabet, start_symbol='',output_path="./",t_max_length=10000)
    al.learn()
#
def test_alergia2():
    sequence,alphabet = load_data(get_path(SeqPath.CRAPS))
    al = ALERGIA(0.008, sequence, alphabet, start_symbol="start", output_path="./", t_max_length=1000)
    al.learn()

#
# def test_aalergia():
#     sequence,alphabet = load_data(get_path(SeqPath.CRAPS), symbols_count=50)
#     al = AALERGIA(64, sequence, alphabet, start_symbol="start", output_path="./", t_max_length=1000)
#     al.learn()


if __name__ == '__main__':
    # test_aalergia()
    # test_alergia2()
    test_alergia1()
    # test_print_pdfa()

