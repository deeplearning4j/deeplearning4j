package org.deeplearning4j.optimizer.listener;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.junit.Assert.assertNotEquals;

/**
 * @author Adam Gibson
 */
public class AssertWeightsDifferentIerationListener implements IterationListener {
    private INDArray lastWeights;
    private String param;
    //forgive 1 time being equal; it could be converged
    private int count = 0;

    public AssertWeightsDifferentIerationListener(String param) {
        this.param = param;
    }

    @Override
    public boolean invoked() {
        return false;
    }

    @Override
    public void invoke() {

    }

    @Override
    public void iterationDone(Model model, int iteration) {
        if (lastWeights == null)
            lastWeights = model.getParam(param).dup();
        else {
            count++;

            if (count < 2)
                return;

            if (count > 2)
                assertNotEquals(lastWeights, model.getParam(param));
            lastWeights = model.getParam(param).dup();
        }
    }
}
