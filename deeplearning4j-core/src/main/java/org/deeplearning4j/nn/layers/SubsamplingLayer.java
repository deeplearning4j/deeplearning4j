package org.deeplearning4j.nn.layers;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Sub sampling layer
 *
 * @author Adam Gibson
 */
public class SubsamplingLayer extends BaseLayer {


    private INDArray featureMap;



    public SubsamplingLayer(NeuralNetConfiguration conf, INDArray W, INDArray b, INDArray input) {
        super(conf, W, b, input);
    }

    @Override
    protected INDArray createBias() {
        return Nd4j.create(conf.getNumFeatureMaps(),1);
    }

    @Override
    protected INDArray createWeightMatrix() {
        return null;
    }

    @Override
    public INDArray activate(INDArray input) {
        for(int i = 0; i < conf.getNumInFeatureMaps(); i++) {

        }
         return super.activate(input);
    }
}
