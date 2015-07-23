package org.deeplearning4j.nn.multilayer;

import static org.junit.Assert.*;

import java.util.Map;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.override.ClassifierOverride;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.layers.recurrent.GravesLSTM;
import org.deeplearning4j.nn.params.GravesLSTMParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class MultiLayerTestRNN {

    @Test
    public void testGravesLSTMInit(){
        int nIn = 8;
        int nOut = 25;
        int nHiddenUnits = 17;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new org.deeplearning4j.nn.conf.layers.GravesLSTM())
                .nIn(nIn).nOut(nOut)
                .activationFunction("tanh")
                .list(2).hiddenLayerSizes(nHiddenUnits)
                .override(1, new ClassifierOverride())
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        //Ensure that we have the correct number weights and biases, that these have correct shape etc.
        Layer layer = network.getLayer(0);
        assertTrue(layer instanceof GravesLSTM);

        Map<String,INDArray> paramTable = layer.paramTable();
        assertTrue(paramTable.size() == 3);	//2 sets of weights, 1 set of biases

        INDArray recurrentWeights = paramTable.get(GravesLSTMParamInitializer.RECURRENT_WEIGHTS);
        assertArrayEquals(recurrentWeights.shape(),new int[]{nHiddenUnits,4*nHiddenUnits+3});	//Should be shape: [layerSize,4*layerSize+3]
        INDArray inputWeights = paramTable.get(GravesLSTMParamInitializer.INPUT_WEIGHTS);
        assertArrayEquals(inputWeights.shape(),new int[]{nIn,4*nHiddenUnits}); //Should be shape: [nIn,4*layerSize]
        INDArray biases = paramTable.get(GravesLSTMParamInitializer.BIAS);
        assertArrayEquals(biases.shape(),new int[]{1,4*nHiddenUnits});	//Should be shape: [1,4*layerSize]

        //Want forget gate biases to be initialized to > 0. See parameter initializer for details
        INDArray forgetGateBiases = biases.get(new NDArrayIndex[]{NDArrayIndex.interval(nHiddenUnits, 2*nHiddenUnits),new NDArrayIndex(0)});
        assertTrue(forgetGateBiases.gt(0).sum(0).getDouble(0)==nHiddenUnits);

        int nParams = recurrentWeights.length() + inputWeights.length() + biases.length();
        assertTrue(nParams == layer.numParams());
    }

    @Test
    public void testGravesTLSTMInitStacked(){
        int nIn = 8;
        int nOut = 25;
        int[] nHiddenUnits = {17,19,23};
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new org.deeplearning4j.nn.conf.layers.GravesLSTM())
                .nIn(nIn).nOut(nOut)
                .activationFunction("tanh")
                .list(nHiddenUnits.length+1).hiddenLayerSizes(nHiddenUnits)
                .override(nHiddenUnits.length, new ClassifierOverride())
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        //Ensure that we have the correct number weights and biases, that these have correct shape etc. for each layer
        for( int i=0; i<nHiddenUnits.length; i++ ){
            Layer layer = network.getLayer(i);
            assertTrue(layer instanceof GravesLSTM);

            Map<String,INDArray> paramTable = layer.paramTable();
            assertTrue(paramTable.size() == 3);	//2 sets of weights, 1 set of biases

            int layerNIn = (i==0 ? nIn : nHiddenUnits[i-1] );

            INDArray recurrentWeights = paramTable.get(GravesLSTMParamInitializer.RECURRENT_WEIGHTS);
            assertArrayEquals(recurrentWeights.shape(),new int[]{nHiddenUnits[i],4*nHiddenUnits[i]+3});	//Should be shape: [layerSize,4*layerSize+3]
            INDArray inputWeights = paramTable.get(GravesLSTMParamInitializer.INPUT_WEIGHTS);
            assertArrayEquals(inputWeights.shape(),new int[]{layerNIn,4*nHiddenUnits[i]}); //Should be shape: [nIn,4*layerSize]
            INDArray biases = paramTable.get(GravesLSTMParamInitializer.BIAS);
            assertArrayEquals(biases.shape(),new int[]{1,4*nHiddenUnits[i]});	//Should be shape: [1,4*layerSize]

            //Want forget gate biases to be initialized to > 0. See parameter initializer for details
            INDArray forgetGateBiases = biases.get(new NDArrayIndex[]{NDArrayIndex.interval(nHiddenUnits[i], 2*nHiddenUnits[i]),new NDArrayIndex(0)});
            assertTrue(forgetGateBiases.gt(0).sum(0).getDouble(0)==nHiddenUnits[i]);

            int nParams = recurrentWeights.length() + inputWeights.length() + biases.length();
            assertTrue(nParams == layer.numParams());
        }
    }


    @Test
    public void testMultiLayerRnn() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(new org.deeplearning4j.nn.conf.layers.GravesLSTM())
                .nIn(10).nOut(10)
                .activationFunction("sigmoid")
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                .weightInit(WeightInit.DISTRIBUTION)
                .list(2)
                .hiddenLayerSizes(10, 10)
                .override(1, new ConfOverride() {
                    @Override
                    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {

                    }
                })
                .backprop(true)
                .pretrain(false)
                .build();


    }


}
