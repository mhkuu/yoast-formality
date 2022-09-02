import numpy as np
from sklearn import tree
from sklearn.tree import _tree


# Converts a decision tree to JavaScript code
# Adapted from https://stackoverflow.com/q/20224526
def tree_to_code(tree: tree, feature_names: list, filename: str = "tree.js", verbose: bool = False):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    code_out = open(filename, 'wt')

    def recurse(node, depth: int):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = round(tree_.threshold[node], 4)
            code_string = f"{indent}if ( {name} <= {threshold} ) {{\n"
            code_out.write(code_string)
            if verbose:
                print(code_string)
            recurse(tree_.children_left[node], depth + 1)

            recurse(tree_.children_right[node], depth + 1)
        else:
            nodeval = tree_.value[node]
            indent2 = "  " * (depth - 1)
            code_string = f"{indent}return \"{'formal' if np.argmax(nodeval) == 0 else 'informal'}\";\n{indent2}}}\n"
            code_out.write(code_string)
            if verbose:
                print(code_string)

    try:
        recurse(0, 1)
    finally:
        code_out.close()
