from graphviz import Digraph
# from aalergia import *

def visulize_tree(tree, name):
    g = Digraph("G", filename=name, format='png', strict=False)
    first_label = list(tree.keys())[0]
    g.node("0", first_label)
    _sub_plot(g, tree, "0")
    g.view()

root = "0"
def _sub_plot(g, tree, inc):
    global root

    first_label = list(tree.keys())[0]
    ts = tree[first_label]
    for i in ts.keys():
        if isinstance(tree[first_label][i], dict):
            root = str(int(root) + 1)
            g.node(root, list(tree[first_label][i].keys())[0])
            g.edge(inc, root, str(i))
            _sub_plot(g, tree[first_label][i], root)
        else:
            root = str(int(root) + 1)
            g.node(root, tree[first_label][i])
            g.edge(inc, root, str(i))



#
# if __name__ == '__main__':
#     s1 = "abab"
#     s2 = "abcdab"
#     s3 = "bacadc"
#     s4 = "acbadc"
#     al = AALERGIA(0.5, [s1, s2,s3,s4])
#     tree = al.fpta()
#     print(al.prefix)
#     # print()
#     d = al.make_tree_data(tree)
#     print(d)
#     visulize_tree(al.make_tree_data(tree), "temp.gv")