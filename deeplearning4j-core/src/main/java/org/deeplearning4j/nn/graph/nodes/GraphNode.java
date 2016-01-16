package org.deeplearning4j.nn.graph.nodes;

import org.nd4j.linalg.api.ndarray.INDArray;

/** A GraphNode is a component of a ComputationGraph, that handles the interactions between layers
 *  */
public interface GraphNode {

    INDArray forward(INDArray... activations);


    INDArray[] backward(INDArray epsilon);


}
