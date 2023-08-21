import os
import argparse
from ete3 import Tree
import pandas as pd
import numpy as np
from libraries import format_pop


def prune_tree(input_tree: Tree, list_taxa: list) -> Tree:
    tree = input_tree.copy()

    # Prune tree
    tree.prune(list_taxa, preserve_branch_length=True)
    assert len(tree.get_leaves()) == len(list_taxa), f"Pruning failed: {len(tree.get_leaves())} != {len(list_taxa)}"

    # Add polytomies if branch length are 0
    remove_nodes = set([n for n in tree.traverse() if (n.dist == 0.0 and not n.is_root())])
    for n in remove_nodes:
        n.delete()
    assert len(set([n for n in tree.traverse()]).intersection(remove_nodes)) == 0
    for n in tree.traverse():
        if not n.is_root():
            assert n.dist > 0.0, f"Branch length is 0.0 for node {n.name}"
    return tree


def sister_species(tree: Tree, specie: str, populations: list) -> Tree:
    leaves = tree.get_leaves_by_name(specie)
    assert len(leaves) == 1
    node = leaves[0]

    while True:
        node = node.up
        leaves = node.get_leaf_names()
        diff_genus_names = [k for k in leaves if k.split("_")[0] != specie.split("_")[0]]
        if len(diff_genus_names) > 0:
            break

    # Compute the distance between the node and the leaves
    distances = []
    for c in node.get_leaves():
        d = node.get_distance(c)
        distances.append(d)
    assert len(distances) == len(node.get_leaves())
    # Assert that the distance are similar
    d = np.mean(distances)
    if not np.allclose(distances, np.mean(distances), atol=1e-5):
        print("Warning: distances are not similar")
        print(distances)
        print(node)

    set_leaves = set(tree.get_leaf_names())
    for pop in populations:
        assert pop not in set_leaves
        node.add_child(name=pop, dist=d)
    return tree


def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df_sample = pd.read_csv(args.sample, sep="\t")
    tree = Tree(args.tree, quoted_node_names=True, format=1)

    for species, df_sub in df_sample.groupby("Species"):
        print(species)
        tree = sister_species(tree, species, [format_pop(p.replace("_", " ")) for p in df_sub["SampleName"]])

    tree = prune_tree(tree, list_taxa=[format_pop(p.replace("_", " ")) for p in df_sample["SampleName"]])
    tree.write(outfile=args.output, format=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tree', required=True, type=str, dest="tree", help="The fasta folder path")
    parser.add_argument('--sample', required=True, type=str, dest="sample", help="The species and pop")
    parser.add_argument('--output', required=True, type=str, dest="output", help="Output tsv path")
    main(parser.parse_args())
