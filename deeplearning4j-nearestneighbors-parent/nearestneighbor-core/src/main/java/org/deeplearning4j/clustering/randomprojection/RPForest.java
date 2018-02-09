package org.deeplearning4j.clustering.randomprojection;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

public class RPForest {

    private int numTrees;
    private List<RPTree> trees;
    private INDArray data;
    private int maxSize = 1000;
    private String similarityFunction;

    public RPForest(int numTrees,int maxSize,String similarityFunction) {
        this.numTrees = numTrees;
        this.maxSize = maxSize;
        this.similarityFunction = similarityFunction;
        trees = new ArrayList<>(numTrees);

    }


    public void fit(INDArray x) {
        for(int i = 0; i < numTrees; i++) {
            RPTree tree = new RPTree(data.columns(),maxSize,similarityFunction);
            tree.buildTree(x);
            trees.add(tree);
        }
    }

    /**
     * Get all candidates relative to a specific datapoint.
     * @param input
     * @return
     */
    public INDArray getAllCandidates(INDArray input) {
        return RPUtils.getAllCandidates(input,trees,similarityFunction);
    }

    public INDArray queryAll(INDArray toQuery,int n) {
        return RPUtils.queryAll(toQuery,data,trees,n,similarityFunction);
    }


}
