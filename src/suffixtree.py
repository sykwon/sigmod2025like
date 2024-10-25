from tqdm import tqdm
from src.util import max_word_len, merge_positional_lists


def build_tree(db, p_bar=True, pos=False):
    tree = SuffixTree(pos)
    tree.add_strings(db, p_bar)
    return tree


class SuffixTree:
    max_word_len = max_word_len

    def __init__(self, pos=False):
        """ SuffixTree

        Parameters
        ----------
        pos : bool, optional
            whether position is added to id or not, by default False
        """
        self.root = SuffixTreeNode()
        self.pos = pos

    def add_strings(self, strings, p_bar=True):
        """ add strings to this suffix tree

        Parameters
        ----------
        strings : list of str
            data strings
        p_bar : bool, optional
            whether progress bar is shown or not, by default True
        """
        if p_bar:
            id_strings = tqdm(enumerate(strings), total=len(strings))
        else:
            id_strings = enumerate(strings)

        for id, rec in id_strings:
            if self.pos:
                self.add_all_suffixes_pos(rec, id)
            else:
                self.add_all_suffixes(rec, id)

    def add_all_suffixes_pos(self, rec, rid):
        for i in range(len(rec)):
            suffix = rec[i:]
            word = suffix[:self.max_word_len]

            node: SuffixTreeNode = self.root
            for ch in word:
                if ch not in node.children:
                    node.children[ch] = SuffixTreeNode()
                node = node.children[ch]
                if len(node.inv_list) == 0 or node.inv_list[-1][0] != rid:
                    node.inv_list.append([rid, [i]])
                else:
                    node.inv_list[-1][1].append(i)

    def add_all_suffixes(self, rec, rid):
        for i in range(len(rec)):
            suffix = rec[i:]
            # print("suffix:", suffix)
            self.add_string(suffix, rid)

    def add_string(self, rec, rid):
        """ add a string with its id to this suffix tree

        Parameters
        ----------
        rec : str
            a string which is a substring of a data string
        rid : int or tuple
            record id
        """
        node: SuffixTreeNode = self.root
        word = rec[:self.max_word_len]
        for ch in word:
            if ch not in node.children:
                node.children[ch] = SuffixTreeNode()
            node = node.children[ch]
            if len(node.inv_list) == 0 or node.inv_list[-1] != rid:
                node.inv_list.append(rid)

    def find_node(self, rec):
        node = self.root
        for ch in rec:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def find_id(self, query):
        assert self.pos
        ss_list = query.split('%')
        ss_list = list(filter(lambda x: len(x) >= 1, ss_list))

        merged_list = None
        prev_len_ss = None

        for ss in ss_list:
            id_list = self.find_node(ss).inv_list
            if merged_list is None:
                merged_list = id_list
            else:
                merged_list = merge_positional_lists(merged_list, id_list, prev_len_ss)

            prev_len_ss = len(ss)
        return merged_list

    def find_candidate_id(self, query):
        substr_list = query.split('%')
        substr_list = list(filter(lambda x: len(x) >= 1, substr_list))

        candidate_set = None
        for substr in substr_list:
            inv_list = self.find_node(substr).inv_list
            if candidate_set is None:
                candidate_set = set(inv_list)
            else:
                candidate_set = candidate_set.intersection(inv_list)
        return candidate_set

    def printTree(self):
        self.root.printTree()


class SuffixTreeNode:
    def __init__(self):
        self.children = {}
        self.inv_list = []

    def count(self):
        return len(self.inv_list)

    def printTree(self, prefix=""):
        for i, (ch, child) in enumerate(self.children.items()):
            if i == 0:
                prefix_sub = prefix
            else:
                prefix_sub = '-' * len(prefix)
            child: SuffixTreeNode = child
            prefix_sub += ch

            # print(ch, end="")
            if len(child.children) == 0:
                print(prefix_sub)
            child.printTree(prefix_sub)

    def find_node(self, string):
        substr_list = string.split('%')
        substr_list = list(filter(lambda x: len(x) >= 1, substr_list))
        assert len(substr_list) <= 2, (substr_list, f"is not supported")
        node = self.root
        for i, substr in enumerate(substr_list):
            if i == 0:
                for ch in substr:
                    if ch not in node.children:
                        return None
                    node = node.children[ch]
            elif i == 1:
                for ch in substr:
                    if ch not in node.children2:
                        return None
                    node = node.children2[ch]
        return node


if __name__ == "__main__":

    # test 1
    print("start test 1")
    strings = ["abc", "adc"]
    strings = ["abc", "bcde"]
    tree = build_tree(strings)
    tree.printTree()
    print("end test 1")
    print()

    # test
    print("start test 2")
    strings = []
    dataName = "dblpSmall"
    with open(f"../data/{dataName}.txt") as f:
        for line in f.readlines():
            strings.append(line.rstrip())
    max_len = 10
    db, queries, res_dict = main_sub(dataName, query_type, max_len)
    tree = build_tree(strings)
    # tree.printTree()
