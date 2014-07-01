package org.deeplearning4j.text.treeparser;

import org.deeplearning4j.rntn.Tree;
import org.deeplearning4j.text.treeparser.transformer.TreeTransformer;

import java.util.ArrayList;
import java.util.List;

/**
 * Binarizes trees
 * @author Adam Gibson
 */
public class BinarizeTreeTransformer implements TreeTransformer {


    @Override
    public Tree transform(Tree t) {
        if(t.isLeaf() || t.isPreTerminal())
             return t.clone();
        else {
            List<Tree> children = t.children();
            while (children.size() == 1 && !children.get(0).isLeaf()) {
                children = children.get(0).children();
            }

            List<Tree> processedChildren = new ArrayList<>();
            for (Tree child : children) {
                processedChildren.add(transform(child));
            }

            Tree ret = new Tree(t.getTokens());
            ret.setLabel(t.label());
            ret.connect(processedChildren);
            return ret;

        }

    }
}
