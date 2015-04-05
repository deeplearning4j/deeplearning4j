package org.deeplearning4j.nn.conf.preprocessor;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Unit variance operation
 *
 * @author Adma Gibson
 */
public class UnitVariancePrePreProcessor implements InputPreProcessor {
    @Override
    public INDArray preProcess(INDArray input) {
        INDArray columnStds = input.std(0);
        columnStds.addi(Nd4j.EPS_THRESHOLD);
        input.diviRowVector(columnStds);
        return input;

    }
}
