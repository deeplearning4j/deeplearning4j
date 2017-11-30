package org.deeplearning4j.nn.layers.CustomLayer;


import Utils.Utils_dpl4j.CustomParamInitializer.ElementWiseParamInitializer;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

/**
 * created by jingshu
 */
public class ElementWiseMultiplicationLayer_impl extends BaseLayer<ElementWiseMultiplicationLayer> {

    public ElementWiseMultiplicationLayer_impl(NeuralNetConfiguration conf){
        super(conf);
    }

    public ElementWiseMultiplicationLayer_impl(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    @Override
    public Gradient error(INDArray errorSignal) {
        INDArray W = getParam(ElementWiseParamInitializer.WEIGHT_KEY);
        Gradient nextLayerGradient = new DefaultGradient();
        INDArray wErrorSignal = errorSignal.mul(W);
        nextLayerGradient.gradientForVariable().put(ElementWiseParamInitializer.WEIGHT_KEY, wErrorSignal);
        return nextLayerGradient;
    }

    @Override
    public Gradient calcGradient(Gradient layerError, INDArray activation) {
        Gradient ret = new DefaultGradient();
        INDArray weightErrorSignal = layerError.getGradientFor(ElementWiseParamInitializer.WEIGHT_KEY);
        INDArray weightError = weightErrorSignal.mul(activation);
        ret.gradientForVariable().put(ElementWiseParamInitializer.WEIGHT_KEY, weightError);
        INDArray biasGradient = weightError.mean(0);
        ret.gradientForVariable().put(ElementWiseParamInitializer.BIAS_KEY, biasGradient);

        return ret;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        //If this layer is layer L, then epsilon for this layer is ((w^(L+1)*(delta^(L+1))^T))^T (or equivalent)
        INDArray z = preOutput(true); //Note: using preOutput(INDArray) can't be used as this does a setInput(input) and resets the 'appliedDropout' flag
        //INDArray activationDerivative = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getLayer().getActivationFunction(), z).derivative());
        //        INDArray activationDerivative = conf().getLayer().getActivationFn().getGradient(z);
        //        INDArray delta = epsilon.muli(activationDerivative);
        INDArray delta = layerConf().getActivationFn().backprop(z, epsilon).getFirst(); //TODO handle activation function params
//        if activation is a identity function
//        delta = Nd4j.ones(this.layerConf().getNIn());

        if (maskArray != null) {
            applyMask(delta);
        }

        Gradient ret = new DefaultGradient();

//        INDArray weightGrad = Nd4j.zeros( gradientViews.get(ElementWiseParamInitializer.WEIGHT_KEY).shape(),'f'); //f order
        INDArray weightGrad =  gradientViews.get(ElementWiseParamInitializer.WEIGHT_KEY); //f order
//        reset weight gradients
        weightGrad.subi(weightGrad);

        for(int row =0;row<input.rows();row++){
            weightGrad.addi(input.getRow(row).mul(delta.getRow(row)));
        }

//        weightGrad.addi(input.getRow(0).mul(delta));

        INDArray biasGrad = gradientViews.get(ElementWiseParamInitializer.BIAS_KEY);
        delta.sum(biasGrad, 0); //biasGrad is initialized/zeroed first

        ret.gradientForVariable().put(ElementWiseParamInitializer.WEIGHT_KEY, weightGrad);
        ret.gradientForVariable().put(ElementWiseParamInitializer.BIAS_KEY, biasGrad);

        INDArray epsilonNext = Nd4j.zeros(params.get(ElementWiseParamInitializer.WEIGHT_KEY).shape());

        for(int row =0;row<delta.rows();row++){
            epsilonNext.addi(params.get(ElementWiseParamInitializer.WEIGHT_KEY).mul(delta.getRow(row)));
        }

        return new Pair<>(ret, epsilonNext);
    }

/*    @Override
    public void fit(INDArray input) {}*/

    /**
     * Returns true if the layer can be trained in an unsupervised/pretrain manner (VAE, RBMs etc)
     *
     * @return true if the layer can be pretrained (using fit(INDArray), false otherwise
     */
    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    public INDArray preOutput(boolean training) {
        applyDropOutIfNecessary(training);
        INDArray b = getParam(DefaultParamInitializer.BIAS_KEY);
        INDArray W = getParam(DefaultParamInitializer.WEIGHT_KEY);

        //Input validation:
        if ( input.columns() != W.columns()) {

            throw new DL4JInvalidInputException(
                    "Input size (" + input.columns() + " columns; shape = " + Arrays.toString(input.shape())
                            + ") is invalid: does not match layer input size (layer # inputs = "
                            + W.shapeInfoToString() + ") " + layerId());
        }

        if (conf.isUseDropConnect() && training && layerConf().getDropOut() > 0) {
            W = Dropout.applyDropConnect(this, DefaultParamInitializer.WEIGHT_KEY);
        }

//        System.out.println(W.shapeInfoToString());
//        System.out.println(input.shapeInfoToString());

        INDArray ret = Nd4j.zeros(input.rows(),input.columns());

        for(int row = 0; row<input.rows();row++){
            ret.put(new INDArrayIndex[]{NDArrayIndex.point(row), NDArrayIndex.all()},input.getRow(row).mul(W).addRowVector(b));
        }

//        INDArray ret = input.mul(W).addRowVector(b);

        if (maskArray != null) {
            applyMask(ret);
        }

        return ret;
    }


    @Override
    public INDArray activationMean() {
        INDArray b = getParam(DefaultParamInitializer.BIAS_KEY);
        INDArray W = getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray ret = Nd4j.zeros(input.rows(),input.columns());

        for(int row = 0; row<input.rows();row++){
            ret.put(new INDArrayIndex[]{NDArrayIndex.point(row), NDArrayIndex.all()},input.getRow(row).mul(W).addRowVector(b));
        }
        return ret;
    }


}