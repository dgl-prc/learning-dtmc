import copy
from collections import defaultdict
import math
from visulize import print_pdfa


class FPTATree(object):
    def __init__(self, alphabet, prefix, label):
        self.alphabet = alphabet
        self.prefix = prefix
        self.label = label
        self.children = []


ID2LABELS = "id2labels"
ID2CHILDREN = "id2children"
ID2PARENT = "id2parent"
ID2ACTIONS = "id2actions"

class ALERGIA(object):

    def __init__(self, alpha, S):
        '''
        Parameters.
        -----------
        alpha:
        S:list. dataset for training, where each item is an iterable object denoting a sequence.
        '''
        self.alpha = alpha
        self.S = S
        self.alphabet = list(set([word for s in S for word in s]))
        self.alphabet.sort()
        self.alphabet2id = {w: i for i, w in enumerate(self.alphabet)}
        self.id2alphabet = {i: w for i, w in enumerate(self.alphabet)}
        self.prefix = set([s[:i + 1] for s in S for i in range(len(s))])
        self.empty = ''
        self.FPTAStates, self.FPTAStatesAction = self.make_fpta_states()
        self.PDFA_T = self.makePDFA()
        self.T, self.prefix2id, self.id2prefix, self.id2subids, self.id2parent, self.id2actions = self.fpta(
            self.FPTAStates,
            self.FPTAStatesAction)

    def get_fre(self, prefix):
        ''' get frequency of p in whole dataset

        Parameters.
        -------------------
        param prefix: the prefix string for looking up
        Return: int. the frequency of p as prefix in dataset.
        '''
        cnt = 0
        for s in self.S:
            if s.startswith(prefix):
                cnt += 1
        return cnt

    def get_fre_t(self, prefix):
        ''' frequence of <prefix, empty> in S

        Parameters.
        -------------------
        prefix:
        Return:
        '''
        cnt = 0
        for s in self.S:
            if s == prefix:
                cnt += 1
        return cnt

    def make_fpta_states(self):
        '''
          This is a tree T with a state for each s ∈ prefix(S).
          The state s is labeled with the frequencies fT (s, σ) (σ ∈ Σ)
          and fT (s, e), where fT (s, σ) is the number of strings in S
          with prefix sσ, and fT (s, e) is the number of occurrences of
          s in S.

          FPTAStates and FPTAStatesAction only save empty string and valid sigma of alphabet

          Return:
          '''
        FPTAStates = defaultdict(list)
        FPTAStatesAction = defaultdict(list)
        for s in self.prefix:
            FPTAStates[s].append(self.get_fre_t(s))  # <s,e>
            FPTAStatesAction[s].append(self.empty)
            for w in self.alphabet:
                fre = self.get_fre(s + w)  # <s,sigma>
                if fre > 0:
                    FPTAStates[s].append(fre)
                    FPTAStatesAction[s].append(w)
        # the first item denotes the frequency of the empty string itself
        # the value is intentionally set as 0 for the probability calculation.
        # todo: remove 9.
        FPTAStates[''].append(9)
        FPTAStates[''].extend([self.get_fre('' + w) for w in self.alphabet if self.get_fre('' + w) != 0])
        FPTAStatesAction[''].append(self.empty)
        FPTAStatesAction[''].extend([w for w in self.alphabet if self.get_fre('' + w) != 0])
        # todo: change related code.
        return FPTAStates, FPTAStatesAction

    def fpta(self, FPTAStates, FPTAStatesAction):
        ''' make a FPTA tree
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
        node_id = 0
        root_node = FPTATree(self.alphabet, '', FPTAStates[''])
        queue = [root_node]
        prefix2id[''] = node_id
        id2prefix[node_id] = ''
        id2parent[node_id] = -1
        while len(queue) > 0:
            c_node = queue.pop(0)  # current node
            for sigma, fre in zip(FPTAStatesAction[c_node.prefix][1:], c_node.label[1:]):
                assert fre != 0
                prefix = c_node.prefix + sigma
                node_id += 1
                new_node = FPTATree(None, prefix, FPTAStates[prefix])
                prefix2id[prefix] = node_id
                id2prefix[node_id] = prefix
                id2parent[node_id] = prefix2id[c_node.prefix]
                id2children[prefix2id[c_node.prefix]].append(node_id)
                id2actions[prefix2id[c_node.prefix]].append(sigma)
                queue.append(new_node)
                c_node.children.append(new_node)

        return root_node, prefix2id, id2prefix, id2children, id2parent, id2actions

    def makePDFA(self):
        ''' normalize the frequencies of a state
        Return: dict. a normalized FPTAStates
        '''
        pdfa = defaultdict(list)
        for state in self.FPTAStates:
            frequencies = self.FPTAStates[state]
            probability = [fre / sum(frequencies) for fre in frequencies]
            pdfa[state] = probability
        return pdfa

    def get_tp(self, p, sigma):
        ''' get the termination probability of 'prefix' based on pdfa of original tree T
        Parameters.
        ---------------
        p: string. the prefix to handle
        sigma: a word in alphabet
        Return: float. the termination probability
        '''
        idx = self.alphabet.index(sigma)
        return self.PDFA_T[p][idx]

    def break_link(self, child_id, A):
        ''' break the link between child and its parent
        The core operations are 1) delete related record from id2parent 2)delete the record of id2children
        child_id:
        A:
        Return:
        '''
        # we assume that there is at most one parent of blue node.
        parent_id = A["id2parent"][child_id]
        ## remove child from parent_id's chilren
        c_idx = A["id2children"][parent_id].index(child_id)
        action = A["id2actions"][parent_id][c_idx]
        action_label = A[ID2LABELS][parent_id][c_idx+1]
        A[ID2LABELS][parent_id].pop(c_idx+1)
        A[ID2ACTIONS][parent_id].remove(action)
        A[ID2CHILDREN][parent_id].remove(child_id)
        ## remove (child_id, parent_id) from id2parent
        del A["id2parent"][child_id]
        # note that we remove the child and its descendants after merging the labels.
        return parent_id, action, action_label

    def build_link(self, red_id, blue_p_id, action, action_label, A):
        '''  make blue node's parent as the parent of red node.

        The core operations are 1) add related record to id2parent 2)add the related record to id2children.

        Parameters.
        -----------------
        red_id:
        blue_p_id:
        A:
        :return:
        '''

        # let blue_p be the red's parent
        if red_id not in A[ID2CHILDREN][blue_p_id]:
            A[ID2CHILDREN][blue_p_id].append(red_id)
            # it is impossible that the action is already in blue_p's action list.
            A[ID2ACTIONS][blue_p_id].append(action)
            A[ID2LABELS][blue_p_id].append(action_label)
        else:
            assert action in A[ID2ACTIONS][blue_p_id]
        # note the red node may have multiple parents.
        if isinstance(A["id2parent"][red_id], list):
            A["id2parent"][red_id].append(blue_p_id)
        else:
            A["id2parent"][red_id] = [A["id2parent"][red_id], blue_p_id]

    def get_next_node(self, current_id, action, A):
        '''

        Parameters.
        -------------------
        current_id:
        action:
        A:
        Return: the id of next node.
        '''
        actions = A["id2actions"][current_id]
        children = A["id2children"][current_id]
        assert len(actions) == len(children)
        return children[actions.index(action)]

    def merge_labels(self, red_id, blue_id, A):
        '''
        this function only change the label of related states.
        Parameters.
        -----------------------
        red_id:
        blue_p_id:
        A:
        Return:
        '''

        # recursively compare red and blue as well as their descendants.
        red_pre = self.id2prefix[red_id]
        blue_pre = self.id2prefix[blue_id]
        red_actions = A[ID2ACTIONS][red_id] if red_id in A[ID2ACTIONS] else []
        blue_actions = A[ID2ACTIONS][blue_id] if blue_id in A[ID2ACTIONS] else []
        red_labels = A[ID2LABELS][red_id]
        blue_labels = A[ID2LABELS][blue_id]

        # assert len(red_actions) == len(red_labels[]) and len(blue_actions) == len(blue_labels)

        # add blue to red node with empty action. Note the first ele in label always corresponds to the empty action.
        b_fre = blue_labels[0]
        red_labels[0] += b_fre

        # traverse their children
        for w in self.alphabet:
            if w in red_actions and w in blue_actions:
                b_w_idx = blue_actions.index(w) + 1  # note that the first of labels is the result of empty action.
                b_fre = blue_labels[b_w_idx]
                r_w_idx = red_actions.index(w) + 1
                red_labels[r_w_idx] += b_fre
                # traverse their children
                # note that we should find red node's children based on the tree after break and relink
                red_child_id = self.get_next_node(red_id, w, A)
                blue_child_id = self.get_next_node(blue_id, w, A)
                assert blue_child_id == self.prefix2id[blue_pre + w]
                self.merge_labels(red_child_id, blue_child_id, A)
        #################################
        # remove the current blue node
        #################################
        if blue_id in A["id2actions"]:
            del A["id2actions"][blue_id]
        if blue_id in A["id2children"]:
            del A["id2children"][blue_id]
        if blue_id in A["id2parent"]:
            del A["id2parent"][blue_id]

    def merge(self, A, red_id, blue_id):
        '''
        The core for merge is redirect the link and update the label (frequency) of q_r and its descendants
        1.
        Parameters.
        ------------------
        A:
        q_r:
        q_b:
        '''
        # break link between q_b and its parent
        parent_id, action,action_label = self.break_link(blue_id, A)
        # build link  between q_r and q_b's parent.
        self.build_link(red_id, parent_id, action, action_label, A)
        # recursively update the label.
        self.merge_labels(red_id, blue_id, A)

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
        state_labels = A[ID2LABELS]
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

        # show the trans matrix
        print_pdfa(trans_func, trans_wfunc, self.alphabet)

        # todo: visulize the tree

    def get_number_T(self, prefix, A):
        ''' get the number of strings terminating at 'prefix'

        :param prefix:
        :param A:
        :return:
        '''
        return A[ID2LABELS][self.prefix2id[prefix]][0]

    def get_number_arriving(self, prefix, A):
        ''' get the total number of strings arriving at 'prefix'
        :param prefix:
        :param A:
        :return:
        '''
        return sum(A[ID2LABELS][self.prefix2id[prefix]])

    def get_number_sigma(self, prefix, sigma, A):
        ''' get the number of strings with <prefix,sigma>
        :param prefix:
        :param sigam:
        :param A:
        :return:
        '''
        p_id = self.prefix2id[prefix]
        # return 0 if current state is leaf node.
        if p_id not in A[ID2ACTIONS] and p_id not in A[ID2CHILDREN]:
            return 0
        if sigma not in A[ID2ACTIONS][p_id]:
            return 0
        a_idx = A[ID2ACTIONS][p_id].index(sigma) + 1 # note. the first ele in STATE_LABELS is the result of termination.
        fa = A[ID2LABELS][self.prefix2id[prefix]][a_idx]
        return fa

    def calculate_threshold(self, q_r, q_b, A):
        n_r = self.get_number_arriving(q_r, A)
        n_b = self.get_number_arriving(q_b, A)
        c = (0.5 * math.log2((2. / self.alpha)(1. / (n_r ** 0.5) + 1. / (n_b ** 0.5)))) ** 0.5
        return c

    def compatible(self, A, red_id, blue_id):
        '''
        Parameters.
        -------------
        T:
        q_r: string. prefix of red node
        q_b: string. prefix of blue node
        Return
        '''

        q_r = self.id2prefix[red_id]
        q_b = self.id2prefix[blue_id]
        # todo: change to q-r == ''
        if q_r != '' and q_r[-1] != q_b[-1]:  # here is different from alergia
            return False

        n_r = self.get_number_arriving(q_r, A)
        n_b = self.get_number_arriving(q_b, A)
        # threshold = ((0.5 * math.log10(2. / self.alpha)) ** 0.5) * ((1. / (n_r ** 0.5) + 1. / (n_b ** 0.5)))
        threshold = ((0.5 * math.log(2. / self.alpha)) ) ** 0.5 * ((1. / (n_r ** 0.5) + 1. / (n_b ** 0.5)))
        ft_r = self.get_number_T(q_r, A)
        ft_b = self.get_number_T(q_b, A)

        diff = abs(ft_r / n_r - ft_b / n_b)
        if diff >= threshold:  # different
            return False
        for sigma in self.alphabet:
            fa_r = self.get_number_sigma(q_r, sigma, A)
            fa_b = self.get_number_sigma(q_b, sigma, A)
            diff = abs(fa_r / n_r - fa_b / n_b)
            if diff >= threshold:  # different
                return False
            if fa_r*fa_b != 0:
                child_red =  A[ID2CHILDREN][red_id][A[ID2ACTIONS][red_id].index(sigma)]
                child_blue = A[ID2CHILDREN][blue_id][A[ID2ACTIONS][blue_id].index(sigma)]
                if not self.compatible(A, child_red, child_blue):
                    return False
        return True

    def learn(self):
        '''
        :return:
        '''
        id2labels = {self.prefix2id[pre]:self.FPTAStates[pre] for pre in self.FPTAStates}
        A = {ID2CHILDREN: copy.deepcopy(self.id2subids), ID2PARENT: copy.deepcopy(self.id2parent),
             ID2LABELS: id2labels, ID2ACTIONS: copy.deepcopy(self.id2actions)}
        RED = [self.prefix2id[self.empty]]  # add empty string
        BLUE = [self.prefix2id[self.empty + w] for w in self.alphabet if (self.empty + w) in self.prefix]
        BLUE.sort()  # Lexicographic order
        while len(BLUE) != 0:
            blue_id = BLUE[0]
            merged = False
            # try to merge qb with one of the red node
            if not merged:
                for red_id in RED:
                    if self.compatible(A, red_id, blue_id):
                        self.merge(A, red_id, blue_id)
                        merged = True
                        break
            if not merged:
                RED.append(blue_id)
                BLUE.remove(blue_id)
                BLUE.extend(A[ID2CHILDREN][blue_id])
            else:
                # remove q_b and its direct children from BLUE
                BLUE.pop(0)
                children_qb = A[ID2CHILDREN][blue_id]
                for q in children_qb:
                    if q in BLUE:
                        BLUE.remove(q)
            ########
            # debug
            #########
            print("red node:{}, blue node:{}, merged:{}".format(red_id,blue_id, merged))
            self.makeDLMC(A)
        return self.makeDLMC(A)

    def tree_info_show(self, fpta_tree):
        '''
        extract tree data for visualization.
        Parameters.
        ------------
        param fpta_tree:
        Return: dict. graphviz-format data for display.
        '''
        freqs = [freq for freq in fpta_tree.label[1:] if freq != 0]
        alphabet = [w for w, freq in zip(self.alphabet, fpta_tree.label[1:]) if freq != 0]
        if len(fpta_tree.children) == 0:
            return fpta_tree.prefix
        return {fpta_tree.prefix: {w + "{:.4f}".format(freq): self.tree_info_show(child) for w, freq, child in
                                   zip(alphabet, freqs, fpta_tree.children)}}

