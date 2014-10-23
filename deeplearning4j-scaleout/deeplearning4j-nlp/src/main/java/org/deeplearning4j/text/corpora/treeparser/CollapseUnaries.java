package org.deeplearning4j.text.corpora.treeparser;

import org.deeplearning4j.models.rntn.Tree;
import org.deeplearning4j.text.corpora.treeparser.transformer.TreeTransformer;

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
            return tree;
        }

        List<Tree> children = tree.children();
        while(children.size() == 1 && !children.get(0).isLeaf()) {
            children = children.get(0).children();
        }

        List<Tree> processed = new ArrayList<>();
        for(Tree child : children)
            processed.add(transform(child));

        Tree ret = new Tree(tree);
        ret.connect(processed);

        return ret;
    }




}
