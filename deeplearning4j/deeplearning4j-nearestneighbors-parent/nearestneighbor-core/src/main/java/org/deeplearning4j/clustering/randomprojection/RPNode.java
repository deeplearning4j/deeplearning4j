package org.deeplearning4j.clustering.randomprojection;


import lombok.Data;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Future;

@Data
public class RPNode {
    private int depth;
    private RPNode left,right;
    private Future<RPNode> leftFuture,rightFuture;
    private List<Integer> indices;
    private double median;
    private RPTree tree;


    public RPNode(RPTree tree,int depth) {
        this.depth = depth;
        this.tree = tree;
        indices = new ArrayList<>();
    }



}
