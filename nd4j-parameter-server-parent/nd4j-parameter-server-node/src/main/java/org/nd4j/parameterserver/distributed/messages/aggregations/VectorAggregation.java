package org.nd4j.parameterserver.distributed.messages.aggregations;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author raver119@gmail.com
 */
public class VectorAggregation extends BaseAggregation {

    public VectorAggregation(short aggregationWidth, INDArray array) {
        super(aggregationWidth);
        this.payload = array.isView() ? array.dup(array.ordering()) : array;
    }
}
