package org.deeplearning4j.graph.models.embeddings;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by Alex on 11/11/2015.
 */
public interface GraphVectorLookupTable {

    int vectorSize();

    void resetWeights();

    void iterate(int first, int second);

    public INDArray getVector(int idx);


}
