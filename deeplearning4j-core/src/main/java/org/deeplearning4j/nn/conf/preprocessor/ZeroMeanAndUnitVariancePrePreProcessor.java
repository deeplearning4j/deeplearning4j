package org.deeplearning4j.nn.conf.preprocessor;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Zero mean and unit variance operation
 *
 * @author Adam Gibson
 */
public class ZeroMeanAndUnitVariancePrePreProcessor implements InputPreProcessor {
    @Override
    public INDArray preProcess(INDArray input) {
        INDArray columnMeans = input.mean(0);
        INDArray columnStds = input.std(0);
        input.subiRowVector(columnMeans);
        columnStds.addi(Nd4j.EPS_THRESHOLD);
        input.diviRowVector(columnStds);
        return input;
    }
}
