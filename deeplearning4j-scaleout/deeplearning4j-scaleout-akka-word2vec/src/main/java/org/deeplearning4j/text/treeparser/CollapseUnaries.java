package org.deeplearning4j.text.treeparser;

import org.deeplearning4j.rntn.Tree;
import org.deeplearning4j.text.treeparser.transformer.TreeTransformer;

import java.util.ArrayList;
import java.util.List;

/**
 * Collapse unaries such that the
 * tree is only made of preterminals and leaves.
 *
 * @author Adam Gibson
 */
public class CollapseUnaries implements TreeTransformer {
    @Override
    public Tree transform(Tree tree) {
        if (tree.isPreTerminal() || tree.isLeaf()) {
            return tree.clone();
        }

        String label = tree.label();
        List<Tree> children = tree.children();
        while (children.size() == 1 && !children.get(0).isLeaf()) {
            children = children.get(0).children();
        }

        List<Tree> processedChildren =  new ArrayList<>();
        for (Tree child : children) {
            processedChildren.add(transform(child));
        }

        Tree ret = new Tree(tree.getTokens());
        ret.connect(processedChildren);
        ret.setLabel(label);
        return ret;
    }
}
