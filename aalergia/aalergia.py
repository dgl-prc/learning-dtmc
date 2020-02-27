from alergia import *
import numpy as np
from utils.time_util import *
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

    def __init__(self, alpha, S, alphabet, start_symbol, output_path):
        self.empty = ''
        self.BS = ","  # bound symbol
        self.START_LABEL = start_symbol
        self.alpha = alpha
        self.S = S
        assert isinstance(alphabet, set)
        self.alphabet = list(alphabet)
        self.alphabet.sort()
        # if self.START_LABEL in self.alphabet:
        #     self.alphabet.remove(self.START_LABEL)
        self.alphabet2id = {w: i for i, w in enumerate(self.alphabet)}
        self.id2alphabet = {i: w for i, w in enumerate(self.alphabet)}
        # note that each seq should be the list type
        print(current_timestamp(),"extract preifx")
        self.prefix, self.real_used_len = self.extract_prefix(S)
        ##########################################
        # to be consistent with matlab code
        ##########################################
        self.prefix.add(self.empty)
        # FPTAStatesFreq and FPTAStatesAction include empty string
        print(current_timestamp(), "make fpta")
        self.FPTAStatesFreq, self.FPTAStatesAction = self.make_fpta_states()
        self.PDFA_T = self.makePDFA()
        print(current_timestamp())
        self.T, self.prefix2id, self.id2prefix, self.id2subids, self.id2parent, self.id2actions = self.fpta(
            self.FPTAStatesFreq,
            self.FPTAStatesAction)
        self.output_path = output_path

        assert len(self.FPTAStatesFreq) == len(self.prefix)
        assert len(self.prefix2id) == len(self.prefix), "prefix2id:{},prefix:{}".format(len(self.prefix2id),
                                                                                        len(self.prefix))
    def fpta(self, FPTAStatesFreq, FPTAStatesAction):
        ''' make a frequency prefx Tree acceptor
        Return:
        prefix2id: dict.
        id2prefix: dict.
        id2children: dict.
        id2parents: dict. In pdfa, a node may have multiple parents， i.e., it can be reached by multiple nodes.
        id2actions: dict.
        '''
        prefix2id = {}
        id2prefix = {}
        id2parent = {}
        id2children = defaultdict(list)
        id2actions = defaultdict(list)

        empty_node_id = 0
        empty_node = FPTATree(self.alphabet, self.empty, FPTAStatesFreq[self.empty])
        prefix2id[self.empty] = empty_node_id
        id2prefix[empty_node_id] = self.empty
        id2parent[empty_node_id] = -1

        node_id = 1
        start_node = FPTATree(self.alphabet, self.START_LABEL, FPTAStatesFreq[self.START_LABEL])
        prefix2id[self.START_LABEL] = node_id
        id2prefix[node_id] = self.START_LABEL
        id2parent[node_id] = 0

        id2children[prefix2id[empty_node.prefix]].append(node_id)
        id2actions[prefix2id[empty_node.prefix]].append(self.START_LABEL)

        queue = [start_node]
        while len(queue) > 0:
            c_node = queue.pop(0)  # current node
            for sigma, fre in zip(FPTAStatesAction[c_node.prefix][1:], c_node.label[1:]):
                assert fre != 0
                prefix = self.concat(c_node.prefix, sigma)
                if prefix in self.prefix:
                    node_id += 1
                    new_node = FPTATree(None, prefix, FPTAStatesFreq[prefix])
                    prefix2id[prefix] = node_id
                    id2prefix[node_id] = prefix
                    id2parent[node_id] = prefix2id[c_node.prefix]
                    id2children[prefix2id[c_node.prefix]].append(node_id)
                    id2actions[prefix2id[c_node.prefix]].append(sigma)
                    queue.append(new_node)
                    c_node.children.append(new_node)

        return empty_node, prefix2id, id2prefix, id2children, id2parent, id2actions

    def makeDLMC(self, A):
        '''build DLMC according to two functions (matrices): transition function (matrix)
        and transition weight function (matrix) by using 'id2parents', 'id2children','id2actions'.
        Note that:
            1. len(id2parents) == len(id2children) + size(leaf nodes). since the leaf node has no children.
            2. len(id2children) == len(id2actions)

        Parameters.
        -------------
        '''
        # build transition matrix
        trans_func = defaultdict(defaultdict)
        id2children = A[ID2CHILDREN]
        id2actions = A[ID2ACTIONS]
        state_labels = A[ID2FREQ]
        # assert len(id2children)==len(id2actions) and len(id2children) == len(id2parents)
        for id in id2children:
            actions = id2actions[id]
            children = id2children[id]
            for action, child in zip(actions, children):
                trans_func[id][action] = child

        # build transition weight matrix
        trans_wfunc = defaultdict(defaultdict)
        for id in id2children:
            node_labels = state_labels[id]
            actions = id2actions[id]
            children = id2children[id]
            for action, child in zip(actions, children):
                w_idx = actions.index(
                    action) + 1  # note that the first ele in state_labels is the result of empty action.
                fre = node_labels[w_idx]
                trans_wfunc[id][action] = fre
        return dict(trans_func), dict(trans_wfunc)

    def calculate_threshold(self, q_r, q_b, A):
        n_r = self.get_number_arriving(q_r, A)
        n_b = self.get_number_arriving(q_b, A)
        c = (6 * self.alpha * math.log(n_r) / n_r) ** 0.5
        c += (6 * self.alpha * math.log(n_b) / n_b) ** 0.5
        return c

    def last_symbol(self, prefix):
        symbols = prefix.split(self.BS)
        return symbols[-1]

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
        last_red = self.last_symbol(q_r)
        last_blue = self.last_symbol(q_b)
        if last_red != last_blue:  # here is different from alergia
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
        if sigma not in self.FPTAStatesAction[p]:
            return 0.
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

    def merge(self, dffa, red_id, blue_id):

        red = [red_id]
        blue = [blue_id]
        trans_func = dffa[STM]
        trans_wfunc = dffa[STWM]

        #################################################
        # replace all transitions leading to qb  with qr
        ###################################################
        for id in trans_func:
            for sigma in trans_func[id]:
                if trans_func[id][sigma] == blue_id:
                    trans_func[id][sigma] = red_id

        while len(blue) > 0:
            qr = red.pop(0)
            qb = blue.pop(0)
            # merge the labels of qr abd qb
            dffa[FFREQ][qr] = dffa[FFREQ][qr] + dffa[FFREQ][qb]
            if qb not in trans_wfunc:  # qb is leaf node
                break
            else:  # qb is non-leaf node
                if qr in trans_wfunc:  # qr is non-leaf nodes
                    for sigma in self.alphabet:
                        if sigma in trans_wfunc[qr] and sigma in trans_wfunc[qb]:
                            # deal with same transition between qr and qb
                            trans_wfunc[qr][sigma] = trans_wfunc[qr][sigma] + trans_wfunc[qb][sigma]
                            red.append(trans_func[qr][sigma])
                            blue.append(trans_func[qb][sigma])
                        else:
                            # deal with diff transition between qr and qb
                            if sigma not in trans_wfunc[qr] and sigma in trans_wfunc[qb]:
                                trans_wfunc[qr][sigma] = trans_wfunc[qb][sigma]
                                trans_func[qr][sigma] = trans_func[qb][sigma]
                else:  # qr is leaf node
                    trans_wfunc[qr] = {}
                    trans_func[qr] = {}
                    for sigma in trans_wfunc[qb]:
                        trans_wfunc[qr][sigma] = trans_wfunc[qb][sigma]
                        trans_func[qr][sigma] = trans_func[qb][sigma]

    def get_children(self, trans_func, prefix_id):
        children = []
        if prefix_id in trans_func.keys():
            for sigma in trans_func[prefix_id]:
                children.append(trans_func[prefix_id][sigma])
        return children

    def learn(self):
        '''
        :return:
        '''
        id2freq = {self.prefix2id[pre]: self.FPTAStatesFreq[pre] for pre in self.FPTAStatesFreq}
        A = {ID2CHILDREN: copy.deepcopy(self.id2subids), ID2PARENT: copy.deepcopy(self.id2parent),
             ID2FREQ: id2freq, ID2ACTIONS: copy.deepcopy(self.id2actions)}
        trans_func, trans_wfunc = self.makeDLMC(A)
        # self.pretty_look(ori_trans_func, ori_trans_wfunc)
        dffa = {}
        dffa[STM] = trans_func
        dffa[STWM] = trans_wfunc
        dffa[FFREQ] = {self.prefix2id[pre]: self.FPTAStatesFreq[pre][0] for pre in self.FPTAStatesFreq}
        dffa[ID2FREQ] = id2freq
        dffa[IDMERGED] = []

        ORI_T = copy.deepcopy(A)  # original tree
        RED = [self.prefix2id[self.START_LABEL]]  # add empty string
        BLUE = [self.prefix2id[self.concat(self.START_LABEL, w)] for w in self.alphabet if
                (self.concat(self.START_LABEL, w)) in self.prefix]
        BLUE.sort()  # Lexicographic order
        itr_cnt = 1
        while len(BLUE) != 0:
            # debug
            print("=======iter:{}=====".format(itr_cnt))
            print("RED:{}".format(RED))
            print("BLUE:{}".format(BLUE))

            itr_cnt += 1
            blue_id = BLUE.pop(0)
            merged = False
            # try to merge qb with one of the red node
            for red_id in RED:
                if self.compatible(ORI_T, red_id, blue_id):
                    self.merge(dffa, red_id, blue_id)
                    dffa[IDMERGED].append(blue_id)
                    print("merge {} ---> {}".format(blue_id, red_id))
                    merged = True
                    # self.pretty_look(dffa[STM],dffa[STWM])
                    break
            if not merged:
                RED.append(blue_id)
            #     assert np.array([int(cid not in BLUE and cid not in RED) for cid in A[ID2CHILDREN][blue_id]]).all()
            #     BLUE.extend(A[ID2CHILDREN][blue_id])
            # if merged:
                # must re-fetch the children of red nodes
            new_RED_children = []
            for red_id in RED:
                new_RED_children.extend(self.get_children(trans_func, red_id))
            for child_id in new_RED_children:
                if child_id not in list(RED + BLUE):
                    BLUE.append(child_id)

        new_trans_func = {}
        new_trans_wfunc = {}
        self.pretty_look(dffa[STM], dffa[STWM])
        # note the leaf node which is not in the trans_func
        for id in RED:
            if id in dffa[STM]:
                new_trans_func[id] = dffa[STM][id]
                new_trans_wfunc[id] = dffa[STWM][id]
            else:
                new_trans_func[id] = {}
                new_trans_wfunc[id] = {}
        print("learned done! final model size:{}".format(len(RED)))
        dffa[STM] = new_trans_func
        dffa[STWM] = new_trans_wfunc
        return dffa

    def pretty_look(self, trans_func, trans_wfunc):

        new_trans_func = np.zeros((len(self.prefix), len(self.alphabet)), dtype=int) + -1
        for prefix_id in trans_func:
            for symbol_id in range(len(self.alphabet)):
                symbol = self.id2alphabet[symbol_id]
                if symbol in trans_func[prefix_id]:
                    new_trans_func[prefix_id][symbol_id] = trans_func[prefix_id][symbol]
                else:
                    new_trans_func[prefix_id][symbol_id] = -1
        new_trans_wfunc = np.zeros((len(self.prefix), len(self.alphabet)), dtype=int)
        for prefix_id in trans_wfunc:
            for symbol_id in range(len(self.alphabet)):
                symbol = self.id2alphabet[symbol_id]
                if symbol in trans_wfunc[prefix_id]:
                    new_trans_wfunc[prefix_id][symbol_id] = trans_wfunc[prefix_id][symbol]
                else:
                    new_trans_wfunc[prefix_id][symbol_id] = 0
        # print(self.id2prefix[7])
        # print(self.id2prefix[33])
        # print(self.id2prefix[47])
        # for s in self.S:
        #     print(s)
        # new_trans_func += 1
        return new_trans_func, new_trans_wfunc

    def _get_valid_states(self, trans_func):
        valid_states = set()
        for s_id in trans_func:
            valid_states.add(s_id)
            for sigma in trans_func[s_id]:
                valid_states.add(trans_func[s_id][sigma])
        return valid_states

    def output_prism(self, dffa, model_name):

        trans_func = dffa[STM]
        trans_wfunc = dffa[STWM]
        self.pretty_look(trans_func, trans_wfunc)
        pm_file = "{}{}".format(model_name, self.real_used_len)
        pm_path = os.path.join(self.output_path, pm_file + ".pm")
        trans_func_path = os.path.join(self.output_path, pm_file + "_transfunc" + ".pkl")

        valid_states = self._get_valid_states(trans_func)
        total_states = len(valid_states)
        id2newId = {}
        old_ids = list(valid_states)
        old_ids.sort(key=lambda x: int(x))
        for new_id, old_id in enumerate(old_ids):
            id2newId[old_id] = new_id + 1  # 1-index

        lines = []
        lines.append("dtmc\n")
        lines.append("\n")
        lines.append("module {}\n".format(pm_file))
        lines.append("s:[1..{}] init 1;\n".format(total_states))
        for start_s in valid_states:
            t = sum([trans_wfunc[start_s][sigma] for sigma in trans_wfunc[start_s]])
            trans_p_info = []
            if len(trans_func[start_s]) == 0:
                trans_p_info.append("{}:(s'={})".format(1, id2newId[start_s]))
            else:
                for sigma in trans_func[start_s]:
                    next_s = trans_func[start_s][sigma]
                    fre = trans_wfunc[start_s][sigma]
                    new_next_id = id2newId[next_s]
                    p = fre / t
                    trans_p_info.append("{}:(s'={})".format(p, new_next_id))
            new_c_id = id2newId[start_s]  # new current id
            lines.append("[]s={} -> {};\n".format(new_c_id, " + ".join(trans_p_info)))
        lines.append("endmodule\n")
        lines.append("\n")  # empty line

        # make labels
        label2newIds = defaultdict(list)
        for id in valid_states:
            prefix = self.id2prefix[id]
            new_id = id2newId[id]
            label = self.last_symbol(prefix)
            label2newIds[label].append(new_id)
        for label in self.alphabet:
            if label in label2newIds:
                ids = label2newIds[label]
                ids.sort()
                head = "label \"{}\" ".format(label)
                tail = "|".join(["s={}".format(s_id) for s_id in ids])
                lines.append("{} = {};\n".format(head, tail))
        with open(pm_path, "wt") as f:
            f.writelines(lines)

        new_trans_func = defaultdict(defaultdict)
        for old_id in trans_func:
            new_id = id2newId[old_id]
            for sigma in trans_func[old_id]:
                next_id = trans_func[old_id][sigma]
                new_next_id = id2newId[next_id]
                new_trans_func[new_id][sigma] = new_next_id
        with open(trans_func_path, "wb") as f:
            pickle.dump(new_trans_func, f)

        print("total state:{}".format(total_states))
        print("Prism model saved in {}".format(pm_path))
        print("Transition Function saved in {}".format(trans_func_path))
