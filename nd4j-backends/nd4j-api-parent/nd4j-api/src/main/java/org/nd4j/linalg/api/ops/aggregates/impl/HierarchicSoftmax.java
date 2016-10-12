package org.nd4j.linalg.api.ops.aggregates.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;
import org.nd4j.linalg.api.ops.aggregates.BaseAggregate;

/**
 * This Op describes HS round for SkipGram/CBOW Hierarchic Softmax
 *
 * @author raver119@gmail.com
 */
public class HierarchicSoftmax extends BaseAggregate {

    public HierarchicSoftmax(INDArray syn0, INDArray syn1, INDArray expTable, int idxSyn0, int idxSyn1, double lr) {

    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String name() {
        return "hs";
    }
}
