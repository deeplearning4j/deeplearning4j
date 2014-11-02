package org.deeplearning4j.optimize.stepfunctions;

import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.optimize.api.StepFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Step with back prop
 *
 * @author Adam Gibson
 */
public class BackPropStepFunction  implements StepFunction {

    private BaseMultiLayerNetwork network;

    public BackPropStepFunction(BaseMultiLayerNetwork network) {
        this.network = network;
    }

    @Override
    public void step(INDArray x, INDArray line, Object[] params) {
        network.backPropStep();
    }

    @Override
    public void step(INDArray x, INDArray line) {
        step();
    }

    @Override
    public void step() {
         network.backPropStep();
    }
}
