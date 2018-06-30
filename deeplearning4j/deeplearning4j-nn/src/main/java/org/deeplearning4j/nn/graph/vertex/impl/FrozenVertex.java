package org.deeplearning4j.nn.graph.vertex.impl;

import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.graph.vertex.BaseWrapperVertex;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * FrozenVertex is used for the purposes of transfer learning
 * A frozen layers wraps another DL4J GraphVertex within it.
 * During backprop, the FrozenVertex is skipped, and any parameters are not be updated.
 * @author Alex Black
 */
public class FrozenVertex extends BaseWrapperVertex {
    public FrozenVertex(GraphVertex underlying) {
        super(underlying);
    }
}
