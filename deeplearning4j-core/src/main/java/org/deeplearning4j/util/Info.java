package org.deeplearning4j.util;

import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.jblas.DoubleMatrix;

import java.util.List;

/**
 * @author Adam Gibson
 */
public class Info {

    /**
     * Prints the activations for a feed forward
     * @param input
     * @param network
     * @return
     */
    public static String activationsFor(DoubleMatrix input,BaseMultiLayerNetwork network) {
        List<DoubleMatrix> activations =  network.feedForward(input);
        StringBuffer sb = new StringBuffer();
        for(int i = 0; i < activations.size(); i++) {
            if(i > 0 && i < activations.size() - 1)
                sb.append("\n===============Activation " + i + "=========================\n");
            else if(i < 1)
                sb.append("\n===============Input=========================\n");
            else if(i >= activations.size() - 1)
                sb.append("\n===============Prediction=========================\n");

            sb.append(activations.get(i).toString().replaceAll(";","\n"));
            sb.append("\n======================================================\n");

        }

        return sb.toString();
    }
}
