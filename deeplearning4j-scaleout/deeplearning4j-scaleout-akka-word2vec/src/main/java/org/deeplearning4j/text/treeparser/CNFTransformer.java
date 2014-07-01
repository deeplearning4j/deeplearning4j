package org.deeplearning4j.text.treeparser;

import org.deeplearning4j.rntn.Tree;
import org.deeplearning4j.text.treeparser.transformer.TreeTransformer;

import java.util.ArrayList;
import java.util.List;

/**
 * Implements CNF binary tree transforms
 * @author Adam Gibson
 */
public class CNFTransformer implements TreeTransformer {

    private int horzMarkov = 999;
    private int vertMarkov = 0;


    @Override
    public Tree transform(Tree t) {
        List<Tree> nodeList = new ArrayList<>();
        nodeList.add(t);
        while(!nodeList.isEmpty()) {
            Tree node = nodeList.remove(0);
            String originalLabel = node.label();
            if(vertMarkov != 0 && !node.equals(t)) {

            }

            for(Tree child : node.children())
                nodeList.add(child);

            if(node.children().size() > 2) {
                
            }



        }
        return null;
    }
}
