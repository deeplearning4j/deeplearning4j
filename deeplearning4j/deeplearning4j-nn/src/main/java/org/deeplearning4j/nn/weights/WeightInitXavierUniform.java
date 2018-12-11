package org.deeplearning4j.nn.weights;

import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * As per Glorot and Bengio 2010: Uniform distribution U(-s,s) with s = sqrt(6/(fanIn + fanOut))
 *
 * @author Adam Gibson
 */
@EqualsAndHashCode
public class WeightInitXavierUniform implements IWeightInit {

    @Override
    public INDArray init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
        //As per Glorot and Bengio 2010: Uniform distribution U(-s,s) with s = sqrt(6/(fanIn + fanOut))
        //Eq 16: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
        double s = Math.sqrt(6.0) / Math.sqrt(fanIn + fanOut);
        Nd4j.rand(paramView, Nd4j.getDistributions().createUniform(-s, s));
        return paramView.reshape(order, shape);
    }
}
