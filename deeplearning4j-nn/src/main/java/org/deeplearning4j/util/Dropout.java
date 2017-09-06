package org.deeplearning4j.util;

import org.deeplearning4j.nn.api.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.LegacyDropOut;
import org.nd4j.linalg.api.ops.impl.transforms.LegacyDropOutInverted;
import org.nd4j.linalg.api.ops.random.impl.DropOut;
import org.nd4j.linalg.api.ops.random.impl.DropOutInverted;
import org.nd4j.linalg.factory.Nd4j;


/**
 * @author Adam Gibson
 */
@Deprecated
public class Dropout {

    private Dropout() {}

    /**
     * Apply drop connect to the given variable
     * @param layer the layer with the variables
     * @param variable the variable to apply
     * @return the post applied drop connect
     */
    @Deprecated
    public static INDArray applyDropConnect(Layer layer, String variable) {
        double dropConnect = 0.5;       //TODO
        INDArray result = layer.getParam(variable).dup();
        if (Nd4j.getRandom().getStatePointer() != null) {
            Nd4j.getExecutioner().exec(new DropOut(result, result, dropConnect));
        } else {
            Nd4j.getExecutioner().exec(new LegacyDropOut(result, result, dropConnect));
        }
        return result;
    }


}
