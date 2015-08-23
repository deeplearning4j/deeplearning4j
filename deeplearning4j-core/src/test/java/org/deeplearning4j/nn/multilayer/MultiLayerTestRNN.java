package org.deeplearning4j.nn.multilayer;

import static org.junit.Assert.*;

import java.util.Map;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.layers.recurrent.GRU;
import org.deeplearning4j.nn.layers.recurrent.GravesLSTM;
import org.deeplearning4j.nn.params.GRUParamInitializer;
import org.deeplearning4j.nn.params.GravesLSTMParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class MultiLayerTestRNN {

    @Test
    public void testGravesLSTMInit(){
        int nIn = 8;
        int nOut = 25;
        int nHiddenUnits = 17;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list(2)
                .layer(0, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder()
                        .nIn(nIn).nOut(nHiddenUnits).weightInit(WeightInit.DISTRIBUTION).activation("tanh").build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                        .nIn(nHiddenUnits).nOut(nOut).weightInit(WeightInit.DISTRIBUTION).activation("tanh").build())
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        //Ensure that we have the correct number weights and biases, that these have correct shape etc.
        Layer layer = network.getLayer(0);
        assertTrue(layer instanceof GravesLSTM);

        Map<String,INDArray> paramTable = layer.paramTable();
        assertTrue(paramTable.size() == 3);	//2 sets of weights, 1 set of biases

        INDArray recurrentWeights = paramTable.get(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY);
        assertArrayEquals(recurrentWeights.shape(),new int[]{nHiddenUnits,4*nHiddenUnits+3});	//Should be shape: [layerSize,4*layerSize+3]
        INDArray inputWeights = paramTable.get(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY);
        assertArrayEquals(inputWeights.shape(),new int[]{nIn,4*nHiddenUnits}); //Should be shape: [nIn,4*layerSize]
        INDArray biases = paramTable.get(GravesLSTMParamInitializer.BIAS_KEY);
        assertArrayEquals(biases.shape(),new int[]{1,4*nHiddenUnits});	//Should be shape: [1,4*layerSize]

        //Want forget gate biases to be initialized to > 0. See parameter initializer for details
        INDArray forgetGateBiases = biases.get(new INDArrayIndex[]{NDArrayIndex.interval(nHiddenUnits, 2 * nHiddenUnits)});
        assertTrue(forgetGateBiases.gt(0).sum(Integer.MAX_VALUE).getDouble(0) == nHiddenUnits);

        int nParams = recurrentWeights.length() + inputWeights.length() + biases.length();
        assertTrue(nParams == layer.numParams());
    }

    @Test
    public void testGravesTLSTMInitStacked() {
        int nIn = 8;
        int nOut = 25;
        int[] nHiddenUnits = {17,19,23};
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list(4)
                .layer(0, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder()
                        .nIn(nIn).nOut(17).weightInit(WeightInit.DISTRIBUTION).activation("tanh").build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder()
                        .nIn(17).nOut(19).weightInit(WeightInit.DISTRIBUTION).activation("tanh").build())
                .layer(2, new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder()
                        .nIn(19).nOut(23).weightInit(WeightInit.DISTRIBUTION).activation("tanh").build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                        .nIn(23).nOut(nOut).weightInit(WeightInit.DISTRIBUTION).activation("tanh").build())
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        //Ensure that we have the correct number weights and biases, that these have correct shape etc. for each layer
        for( int i=0; i<nHiddenUnits.length; i++ ){
            Layer layer = network.getLayer(i);
            assertTrue(layer instanceof GravesLSTM);

            Map<String,INDArray> paramTable = layer.paramTable();
            assertTrue(paramTable.size() == 3);	//2 sets of weights, 1 set of biases

            int layerNIn = (i == 0 ? nIn : nHiddenUnits[i-1] );

            INDArray recurrentWeights = paramTable.get(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY);
            assertArrayEquals(recurrentWeights.shape(),new int[]{nHiddenUnits[i],4*nHiddenUnits[i]+3});	//Should be shape: [layerSize,4*layerSize+3]
            INDArray inputWeights = paramTable.get(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY);
            assertArrayEquals(inputWeights.shape(),new int[]{layerNIn,4*nHiddenUnits[i]}); //Should be shape: [nIn,4*layerSize]
            INDArray biases = paramTable.get(GravesLSTMParamInitializer.BIAS_KEY);
            assertArrayEquals(biases.shape(),new int[]{1,4*nHiddenUnits[i]});	//Should be shape: [1,4*layerSize]

            //Want forget gate biases to be initialized to > 0. See parameter initializer for details
            INDArray forgetGateBiases = biases.get(new INDArrayIndex[]{NDArrayIndex.interval(nHiddenUnits[i], 2 * nHiddenUnits[i])});
            assertTrue(forgetGateBiases.gt(0).sum(Integer.MAX_VALUE).getDouble(0) == nHiddenUnits[i]);

            int nParams = recurrentWeights.length() + inputWeights.length() + biases.length();
            assertTrue(nParams == layer.numParams());
        }
    }
    
    @Test
    public void testGRUInit(){
        int nIn = 8;
        int nOut = 25;
        int nHiddenUnits = 17;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list(2)
                .layer(0, new org.deeplearning4j.nn.conf.layers.GRU.Builder()
                        .nIn(nIn).nOut(nHiddenUnits).weightInit(WeightInit.DISTRIBUTION).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                        .nIn(nHiddenUnits).nOut(nOut).weightInit(WeightInit.DISTRIBUTION).build())
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        //Ensure that we have the correct number weights and biases, that these have correct shape etc.
        Layer layer = network.getLayer(0);
        assertTrue(layer instanceof GRU);

        Map<String,INDArray> paramTable = layer.paramTable();
        assertTrue(paramTable.size() == 3);	//2 sets of weights, 1 set of biases

        INDArray recurrentWeights = paramTable.get(GRUParamInitializer.RECURRENT_WEIGHT_KEY);
        assertArrayEquals(recurrentWeights.shape(),new int[]{nHiddenUnits,3*nHiddenUnits});	//Should be shape: [layerSize,3*layerSize]
        INDArray inputWeights = paramTable.get(GRUParamInitializer.INPUT_WEIGHT_KEY);
        assertArrayEquals(inputWeights.shape(),new int[]{nIn,3*nHiddenUnits}); //Should be shape: [nIn,3*layerSize]
        INDArray biases = paramTable.get(GRUParamInitializer.BIAS_KEY);
        assertArrayEquals(biases.shape(),new int[]{1,3*nHiddenUnits});	//Should be shape: [1,3*layerSize]

        int nParams = recurrentWeights.length() + inputWeights.length() + biases.length();
        assertTrue(nParams == layer.numParams());
    }

    @Test
    public void testGRUInitStacked() {
        int nIn = 8;
        int nOut = 25;
        int[] nHiddenUnits = {17,19,23};
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list(4)
                .layer(0, new org.deeplearning4j.nn.conf.layers.GRU.Builder()
                        .nIn(nIn).nOut(17).weightInit(WeightInit.DISTRIBUTION).activation("tanh").build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.GRU.Builder()
                        .nIn(17).nOut(19).weightInit(WeightInit.DISTRIBUTION).activation("tanh").build())
                .layer(2, new org.deeplearning4j.nn.conf.layers.GRU.Builder()
                        .nIn(19).nOut(23).weightInit(WeightInit.DISTRIBUTION).activation("tanh").build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                		.weightInit(WeightInit.DISTRIBUTION).activation("tanh").nIn(23).nOut(nOut).build())
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        //Ensure that we have the correct number weights and biases, that these have correct shape etc. for each layer
        for( int i=0; i<nHiddenUnits.length; i++ ){
            Layer layer = network.getLayer(i);
            assertTrue(layer instanceof GRU);

            Map<String,INDArray> paramTable = layer.paramTable();
            assertTrue(paramTable.size() == 3);	//2 sets of weights, 1 set of biases

            int layerNIn = (i == 0 ? nIn : nHiddenUnits[i-1] );

            INDArray recurrentWeights = paramTable.get(GRUParamInitializer.RECURRENT_WEIGHT_KEY);
            assertArrayEquals(recurrentWeights.shape(),new int[]{nHiddenUnits[i],3*nHiddenUnits[i]});	//Should be shape: [layerSize,3*layerSize]
            INDArray inputWeights = paramTable.get(GRUParamInitializer.INPUT_WEIGHT_KEY);
            assertArrayEquals(inputWeights.shape(),new int[]{layerNIn,3*nHiddenUnits[i]}); //Should be shape: [nIn,3*layerSize]
            INDArray biases = paramTable.get(GRUParamInitializer.BIAS_KEY);
            assertArrayEquals(biases.shape(),new int[]{1,3*nHiddenUnits[i]});	//Should be shape: [1,3*layerSize]

            int nParams = recurrentWeights.length() + inputWeights.length() + biases.length();
            assertTrue(nParams == layer.numParams());
        }
    }

}
