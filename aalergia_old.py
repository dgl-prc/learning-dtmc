from alergia import *

'''
Note that there exists some difference between alergia and aalergia:
1) how the children of compatible node are treated.
    In alergia, it deletes the subtree of the compatible node, while in aalergia, it adds the compatible node's children
    into blue.

2) how the threshold is calculated.
     In alergia, the threshold depends on the number of strings arriving the red node (n_r) and the blue node (n_b).
     By contrast, the calculation in aalergia is a bit complex.
     Notice that, the threshold could be greater than 1 in aalergia
     
3) how the the number of strings arriving the red node  (n_r) and the blue node (n_b) are calculated.
    In alergia, the two numbers are derived from the current merged tree
    By contrast, the two numbers are derived from the original unmerged tree.
    
4) how the compatibility value is calculated. ----> tobe confirm?
    In alergia, it is the difference between the ration of termination strings and incoming strings of two nodes.
    By contrast, it is the infinite norm between the renormalized subsequent children distribution. 
    (i.e.,  t(q, sigma) <--- t(q, sigma) / (1- t(q, e) )) 

'''


class AALERGIA(ALERGIA):

    def __init__(self, alpha, S, start_symbol, output_path, t_max_length):
        self.empty = ''
        self.BS = ","  # bound symbol
        self.START_LABEL = start_symbol
        self.alpha = alpha
        self.S = S
        self.t_max_size = t_max_length  # accumulated length of used trace
        self.alphabet = list(set([word for s in S for word in s]))
        if self.START_LABEL in self.alphabet:
            self.alphabet.remove(self.START_LABEL)
        self.alphabet.sort()
        self.alphabet2id = {w: i for i, w in enumerate(self.alphabet)}
        self.id2alphabet = {i: w for i, w in enumerate(self.alphabet)}
        # note that each seq should be the list type
        self.prefix, self.real_used_len = self.extract_prefix(S)
        self.FPTAStates, self.FPTAStatesAction = self.make_fpta_states()
        self.PDFA_T = self.makePDFA()
        self.T, self.prefix2id, self.id2prefix, self.id2subids, self.id2parent, self.id2actions = self.fpta(
            self.FPTAStates,
            self.FPTAStatesAction)
        self.output_path = output_path
        assert len(self.prefix2id) == len(self.prefix)

    def calculate_threshold(self, q_r, q_b, A):
        n_r = self.get_number_arriving(q_r, A)
        n_b = self.get_number_arriving(q_b, A)
        c = (6 * self.alpha * math.log(n_r) / n_r) ** 0.5
        c += (6 * self.alpha * math.log(n_b) / n_b) ** 0.5
        return c

    def compatible(self, T, red_id, blue_id):
        '''
        Differences:
            1. AALERGIA dermines the compatibility via the original tree instead of the current automata
            2.
        :param T:
        :param red_id:
        :param blue_id:
        :return:
        '''
        q_r = self.id2prefix[red_id]
        q_b = self.id2prefix[blue_id]
        if q_r != self.START_LABEL and q_r[-1] == q_b[-1]:  # here is different from alergia
            return False
        threshold = self.calculate_threshold(q_r, q_b, T)
        return self.compatible_recurse(T, q_r, q_b, 1, 1, threshold)

    def get_tp(self, p, sigma):
        ''' get the termination probability of 'prefix' based on pdfa of original tree T
        Parameters.
        ---------------
        p: string. the prefix to handle
        sigma: a word in alphabet
        Return: float. the termination probability
        '''
        idx = self.FPTAStatesAction[p].index(sigma)
        return self.PDFA_T[p][idx]

    def compatible_recurse(self, T, q_r, q_b, p_r, p_b, eps):

        if p_r <= eps and p_b <= eps:
            return True
        if p_r > eps and p_b == 0:
            return False
        if p_b > eps and q_r == 0:
            return False
        if abs(p_r * self.get_tp(q_r, self.empty) - p_b * self.get_tp(q_b, self.empty)) > eps:
            return False
        for sigma in self.alphabet:
            qr_s = self.concat(q_r, sigma)
            qb_s = self.concat(q_b, sigma)
            # if qb_s in self.prefix and qr_s in self.prefix:
            if qb_s in self.prefix or qr_s in self.prefix:
                if not self.compatible_recurse(T,
                                               qr_s,
                                               qb_s,
                                               p_r * self.get_tp(q_r, sigma),
                                               p_b * self.get_tp(q_b, sigma),
                                               eps):
                    return False
        return True

    def learn(self):
        '''
        :return:
        '''
        id2labels = {self.prefix2id[pre]: self.FPTAStates[pre] for pre in self.FPTAStates}
        A = {ID2CHILDREN: copy.deepcopy(self.id2subids), ID2PARENT: copy.deepcopy(self.id2parent),
             ID2LABELS: id2labels, ID2ACTIONS: copy.deepcopy(self.id2actions)}
        ORI_T = copy.deepcopy(A)  # original tree
        RED = [self.prefix2id[self.START_LABEL]]  # add empty string
        BLUE = [self.prefix2id[self.concat(self.START_LABEL, w)] for w in self.alphabet if
                (self.concat(self.START_LABEL, w)) in self.prefix]
        BLUE.sort()  # Lexicographic order
        while len(BLUE) != 0:
            blue_id = BLUE.pop(0)
            merged = False
            # try to merge qb with one of the red node
            if not merged:
                for red_id in RED:
                    if self.compatible(ORI_T, red_id, blue_id):
                        self.merge(A, red_id, blue_id)
                        merged = True
                        break
            if not merged:
                RED.append(blue_id)
            else:
                # add q_b's direct children into BLUE
                children_qb = A[ID2CHILDREN][blue_id]
                for q in children_qb:
                    if q in BLUE:
                        BLUE.append(q)
        return self.makeDLMC(A)

# if __name__ == '__main__':
#     sequence = ["110", '', '', '', '0', '', '00', '00', '', '', '', '10110', '', '', '100']
#     al = AALERGIA(0.8, sequence)
#     al.learn()
#     # tree, prefix2id, id2prefix, id2children, id2parent = al.fpta(al.FPTAStates, al.FPTAStatesAction)
#     # print(al.prefix)
#     # # print()
#     # d = al.tree_info_show(tree)
#     # print(d)
#     # visulize_tree(d, "temp2.gv")
