package org.deeplearning4j.graph.models.embeddings;

import org.nd4j.linalg.api.ndarray.INDArray;

/**Lookup table for vector representations of the vertices in a graph
 */
public interface GraphVectorLookupTable {

    /**The size of the vector representations
     */
    int vectorSize();

    void resetWeights();

    /** Conduct learning given a pair of vertices (in and out) */
    void iterate(int first, int second);

    /** Get the vector for the vertex with index idx */
    public INDArray getVector(int idx);


}
