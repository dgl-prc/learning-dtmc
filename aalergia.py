import copy
from collections import defaultdict
import math
from visulize import visulize_tree


class FPTATree(object):
    def __init__(self, alphabet, prefix, label):
        self.alphabet = alphabet
        self.prefix = prefix
        self.label = label
        self.children = []


class AALERGIA(object):

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

        prefix2id:
        id2prefix:
        id2children:
        id2parents

        Return:
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
        A["id2actions"][parent_id].remove(action)
        A["id2children"][parent_id].remove(child_id)
        ## remove (child_id, parent_id) from id2parent
        del A["id2parent"][child_id]
        return parent_id, action

    def build_link(self, red_id, blue_p_id, action, A):
        '''  choose blue node's parent as the parent of red node.

        The core operations are 1) add related record to id2parent 2)add the related record to id2children.

        Parameters.
        -----------------
        red_id:
        blue_p_id:
        A:
        :return:
        '''

        # let blue_p be the red's parent
        if red_id not in A["id2children"][blue_p_id]:
            A["id2children"][blue_p_id].append(red_id)
            # it is impossible that the action is already in blue_p's action list.
            A["id2actions"][blue_p_id].append(action)
        else:
            assert action in A["id2actions"][blue_p_id]
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
        red_actions = A["id2actions"][red_id]
        blue_actions = A["id2actions"][blue_id]
        red_labels = A["state_labels"][red_pre]
        blue_labels = A["state_labels"][blue_pre]

        # assert len(red_actions) == len(red_labels[]) and len(blue_actions) == len(blue_labels)

        # add blue to red node. Note the first ele in label always corresponds to the empty action.
        b_fre = blue_labels[0]
        red_labels[0] += b_fre

        # traverse their children
        for w in self.alphabet:
            if w in red_actions and w in blue_actions:  #
                b_w_idx = blue_actions.index(w)
                b_fre = blue_labels[b_w_idx]
                r_w_idx = red_actions.index(w)
                red_labels[r_w_idx] += b_fre
                # traverse their children
                # note that we should find red node's children based on the tree after break and relink
                red_child_id = self.get_next_node(red_id, w, A)
                blue_child_id = self.get_next_node(blue_id, w, A)
                assert blue_child_id == self.prefix2id[blue_pre + w]
                self.merge_labels(red_child_id, blue_child_id,A)

    def merge(self, A, q_r, q_b):
        '''
        The core for merge is redirect the link and update the label (frequency) of q_r and its descendants
        1.
        Parameters.
        ------------------
        A:
        q_r:
        q_b:
        '''
        id_qr = self.prefix2id[q_r]
        id_qb = self.prefix2id[q_b]

        # break link between q_b and its parent
        parent_id,action = self.break_link(id_qb, A)
        # build link  between q_r and q_b's parent.
        self.build_link(id_qr, parent_id, action, A)
        # recursively update the label.
        self.merge_labels(id_qr, id_qb, A)

    def makeDLMC(self, ftpa):
        pass

    def calculate_threshold(self, q_r, q_b):
        f_qr = self.get_fre(q_r)
        f_qb = self.get_fre(q_b)
        mu_qr = (6 * self.alpha * math.log2(f_qr) / f_qr) ** 0.5
        mu_qb = (6 * self.alpha * math.log2(f_qb) / f_qb) ** 0.5
        return mu_qr + mu_qb

    def compatible(self, T, q_r, q_b):
        # l_qr = self.FPTAStates[q_r]
        # l_qb = self.FPTAStates[q_b]
        # if " ".join(l_qr) != " ".join(l_qb):  # if the label is not equal
        #     return False

        # todo: change to q-r == ''
        if q_r != '' and q_r[-1] == q_b[-1]:  # here is different from alergia
            return False

        threshold = self.calculate_threshold(q_r, q_b)
        return self.compatible_recurse(T, q_r, q_b, 1, 1, threshold)

    def compatible_recurse(self, T, q_r, q_b, p_r, p_b, eps):

        if p_r < eps and p_b < eps:
            return True
        if p_r > eps and p_b == 0:
            return False
        if p_b > eps and q_r == 0:
            return False
        if abs(p_r * self.get_tp(q_r) - p_b * self.get_tp(q_b)) > eps:
            return False
        for sigma in self.alphabet:
            if not self.compatible_recurse(T,
                                           q_r + sigma,
                                           q_b + sigma,
                                           p_r * self.get_tp(q_r, sigma),
                                           p_b * self.get_tp(q_b, sigma),
                                           eps):
                return False
        return True

    def learn(self):

        T = self.T  # original tree
        A = {"id2children": copy.deepcopy(self.id2subids), "id2parent": copy.deepcopy(self.id2parent),
             "state_labels": copy.deepcopy(self.FPTAStates), "id2actions": copy.deepcopy(self.id2actions)}
        RED = [self.empty]  # add empty string
        BLUE = [self.empty + w for w in self.alphabet if (self.empty + w) in self.prefix]
        BLUE.sort()  # Lexicographic order
        while len(BLUE) != 0:
            q_b = BLUE[0]
            merged = False
            # try to merge qb with one of the red node
            if not merged:
                for q_r in RED:
                    if self.compatible(T, q_r, q_b):
                        self.merge(A, q_r, q_b)
                        merged = True
            if not merged:
                RED.append(q_b)
            else:
                # remove q_b and its direct children from BLUE
                BLUE.pop(0)
                children_qb = [q_b + w for w in self.alphabet if (q_b + w) in self.prefix]
                for q in children_qb:
                    BLUE.remove(q)
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


if __name__ == '__main__':
    sequence = ["110", '', '', '', '0', '', '00', '00', '', '', '', '10110', '', '', '100']
    al = AALERGIA(0.8, sequence)
    al.learn()
    # tree, prefix2id, id2prefix, id2children, id2parent = al.fpta(al.FPTAStates, al.FPTAStatesAction)
    # print(al.prefix)
    # # print()
    # d = al.tree_info_show(tree)
    # print(d)
    # visulize_tree(d, "temp2.gv")
