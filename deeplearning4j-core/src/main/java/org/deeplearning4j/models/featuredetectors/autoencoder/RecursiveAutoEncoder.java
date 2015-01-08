package org.deeplearning4j.models.featuredetectors.autoencoder;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.RecursiveParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 1/7/15.
 */
public class RecursiveAutoEncoder extends BaseLayer {

    public RecursiveAutoEncoder(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public void update(Gradient gradient) {

    }

    @Override
    public double score() {
        return 0;
    }



    private List<INDArray> batches() {
        return new ArrayList<>();
    }


    @Override
    public INDArray transform(INDArray data) {
         return conf.getActivationFunction().apply(input.mmul(params.get(RecursiveParamInitializer.W)).addiRowVector(params.get(RecursiveParamInitializer.BIAS)));
    }


    public INDArray decode(INDArray input) {
        return input.mmul(params.get(RecursiveParamInitializer.U).addiRowVector(params.get(RecursiveParamInitializer.C)));
    }

    @Override
    public void setParams(INDArray params) {

    }

    @Override
    public void iterate(INDArray input) {

    }

    @Override
    public Gradient getGradient() {
        return null;
    }
}
