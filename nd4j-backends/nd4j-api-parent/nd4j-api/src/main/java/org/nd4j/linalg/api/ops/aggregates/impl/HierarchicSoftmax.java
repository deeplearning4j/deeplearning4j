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

    public HierarchicSoftmax(INDArray syn0, INDArray syn1, INDArray expTable, INDArray neu1e, int idxSyn0, int idxSyn1, int code, double lr) {
        arguments.add(syn0);
        arguments.add(syn1);
        arguments.add(expTable);
        arguments.add(neu1e);

        indexingArguments.add(idxSyn0);
        indexingArguments.add(idxSyn1);
        indexingArguments.add(neu1e.length());
        indexingArguments.add(expTable.length());
        indexingArguments.add(code);

        realArguments.add(lr);
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String name() {
        return "aggregate_hs";
    }

    @Override
    public int maxArguments() {
        return 4;
    }

    @Override
    public int maxShapes() {
        return 0;
    }

    @Override
    public int maxIntArrays() {
        return 0;
    }

    @Override
    public int maxIntArraySize() {
        return 0;
    }

    @Override
    public int maxIndexArguments() {
        return 5;
    }

    @Override
    public int maxRealArguments() {
        return 1;
    }
}
