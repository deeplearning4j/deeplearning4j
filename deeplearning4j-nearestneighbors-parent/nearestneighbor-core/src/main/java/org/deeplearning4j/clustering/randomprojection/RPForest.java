package org.deeplearning4j.clustering.randomprojection;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 *
 */
@Data
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


    /**
     * Build the trees from the given dataset
     * @param x the input dataset (should be a 2d matrix)
     */
    public void fit(INDArray x) {
        this.data = x;
        for(int i = 0; i < numTrees; i++) {
            RPTree tree = new RPTree(data.columns(),maxSize,0,similarityFunction);
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

    /**
     * Query results up to length n
     * nearest neighbors
     * @param toQuery the query item
     * @param n the number of nearest neighbors for the given data point
     * @return the indices for the nearest neighbors
     */
    public INDArray queryAll(INDArray toQuery,int n) {
        return RPUtils.queryAll(toQuery,data,trees,n,similarityFunction);
    }


}
