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
            return tree;
        }
        List<Tree> nodeList = new ArrayList<>();
        nodeList.add(tree);
        while(!nodeList.isEmpty()) {
            Tree currentNode = nodeList.remove(0);
            if(currentNode.children().size() == 1) {
                currentNode.setLabel(currentNode.label() + "(" + currentNode.firstChild().label());
                List<Tree> currChildren = new ArrayList<>(currentNode.children());
                currentNode.children().clear();
                currentNode.children().addAll(currChildren.get(0).children());
            }
            else {
              nodeList.addAll(currentNode.children());
            }
        }

        return tree;
    }
}
