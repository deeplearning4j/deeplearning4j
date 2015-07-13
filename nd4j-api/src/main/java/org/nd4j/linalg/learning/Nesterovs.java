package org.nd4j.linalg.learning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;

/**
 * @author Adam Gibson
 */
public class Nesterovs implements Serializable,GradientUpdater {
    private double momentum = 0.5;
    private INDArray lastGradient;


    /**
     * Get the nesterov udpdate
     * @param gradient the gradient to get the update for
     *
     * @return
     */
    @Override
    public INDArray getGradient(INDArray gradient) {
      if(lastGradient == null)
          lastGradient = Nd4j.zeros(gradient.shape());
        INDArray ret  = lastGradient.mul(momentum).subi(gradient);
        lastGradient = ret;
        return ret;
    }


}
