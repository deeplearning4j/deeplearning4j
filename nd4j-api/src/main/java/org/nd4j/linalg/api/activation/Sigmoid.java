package org.nd4j.linalg.api.activation;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.factory.ElementWiseOpFactories;
import org.nd4j.linalg.ops.factory.ElementWiseOpFactory;

/**
 * Sigmoid function (complex AND real!)
 *
 * http://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1742-18.pdf
 *
 * For complex we set the k = 1
 *
 *
 * @author Adam Gibson
 */
public class Sigmoid extends BaseActivationFunction {


    /**
     *
     */
    private static final long serialVersionUID = -6280602270833101092L;
    private int k = 1;



    /**
     * The class used for transformation
     *
     * @return the class used for transformation
     */
    @Override
    public ElementWiseOpFactory transformFactory() {
        return ElementWiseOpFactories.sigmoid();
    }

    @Override
    public INDArray applyDerivative(INDArray input) {
        INDArray rSub = input.rsub(1);
        return input.mul(rSub);
    }



}
