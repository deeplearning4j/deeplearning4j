package org.nd4j.linalg.api.ops.aggregates.impl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.aggregates.BaseAggregate;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This aggregate encapsulates SkipGram trainng round for a given word and context
 *
 * @author raver119@gmail.com
 */
public class SkipGram extends BaseAggregate {


    public SkipGram(INDArray syn0, INDArray syn1, INDArray syn1Neg, INDArray expTable, int idxSyn0, int[] idxSyn1, int[] codes, int negativeRounds, int ngStarter, int vectorLength, double alpha) {
        indexingArguments.add(idxSyn0);
        indexingArguments.add(vectorLength);
        indexingArguments.add(idxSyn1.length);
        indexingArguments.add(negativeRounds);
        indexingArguments.add(expTable.length());
        indexingArguments.add(syn0.rows());
        indexingArguments.add(ngStarter);

        arguments.add(syn0);
        arguments.add(syn1);
        arguments.add(expTable);
        arguments.add(syn1Neg);

        shapes.add(Nd4j.getDataBufferFactory().createInt(idxSyn1));
        shapes.add(Nd4j.getDataBufferFactory().createInt(codes));

        realArguments.add(alpha);
    }


    @Override
    public String name() {
        return "aggregate_skipgram";
    }

    @Override
    public int opNum() {
        return 3;
    }
}
