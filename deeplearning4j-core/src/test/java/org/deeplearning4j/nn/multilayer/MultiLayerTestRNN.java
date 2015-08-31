package org.deeplearning4j.nn.multilayer;

import static org.junit.Assert.*;

import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.layers.recurrent.GRU;
import org.deeplearning4j.nn.layers.recurrent.GravesLSTM;
import org.deeplearning4j.nn.params.GRUParamInitializer;
import org.deeplearning4j.nn.params.GravesLSTMParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

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
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
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
                .layer(3, new RnnOutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
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
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
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
                .layer(3, new RnnOutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
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

    @Test
    public void testRnnStateMethods(){
    	Nd4j.getRandom().setSeed(12345);
    	int timeSeriesLength = 6;

    	MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    		.list(3)
    		.layer(0,new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder()
    			.nIn(5).nOut(7).activation("tanh").weightInit(WeightInit.DISTRIBUTION)
    			.dist(new NormalDistribution(0,0.5)).build())
    		.layer(1,new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder()
    			.nIn(7).nOut(8).activation("tanh").weightInit(WeightInit.DISTRIBUTION)
    			.dist(new NormalDistribution(0,0.5)).build())
    		.layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).weightInit(WeightInit.DISTRIBUTION)
    			.nIn(8).nOut(4).activation("softmax").weightInit(WeightInit.DISTRIBUTION)
    			.dist(new NormalDistribution(0,0.5)).build())
    		.build();
    	MultiLayerNetwork mln = new MultiLayerNetwork(conf);

    	INDArray input = Nd4j.rand(new int[]{3,5,timeSeriesLength});

    	List<INDArray> allOutputActivations = mln.feedForward(input, true);
    	INDArray outAct = allOutputActivations.get(3);

    	INDArray outRnnTimeStep = mln.rnnTimeStep(input);

    	assertTrue(outAct.equals(outRnnTimeStep));	//Should be identical here

    	Map<String,INDArray> currStateL0 = mln.rnnGetPreviousState(0);
    	Map<String,INDArray> currStateL1 = mln.rnnGetPreviousState(1);

    	assertTrue(currStateL0.size()==2);
    	assertTrue(currStateL1.size()==2);

    	INDArray lastActL0 = currStateL0.get(GravesLSTM.STATE_KEY_PREV_ACTIVATION);
    	INDArray lastMemL0 = currStateL0.get(GravesLSTM.STATE_KEY_PREV_MEMCELL);
    	assertTrue(lastActL0!=null && lastMemL0!=null);

    	INDArray lastActL1 = currStateL1.get(GravesLSTM.STATE_KEY_PREV_ACTIVATION);
    	INDArray lastMemL1 = currStateL1.get(GravesLSTM.STATE_KEY_PREV_MEMCELL);
    	assertTrue(lastActL1!=null && lastMemL1!=null);

    	INDArray expectedLastActL0 = allOutputActivations.get(1).tensorAlongDimension(timeSeriesLength-1,1,0);
    	assertTrue(expectedLastActL0.equals(lastActL0));

    	INDArray expectedLastActL1 = allOutputActivations.get(2).tensorAlongDimension(timeSeriesLength-1,1,0);
    	assertTrue(expectedLastActL1.equals(lastActL1)); 

    	//Check clearing and setting of state:
    	mln.rnnClearPreviousState();
    	assertTrue(mln.rnnGetPreviousState(0).size()==0);
    	assertTrue(mln.rnnGetPreviousState(1).size()==0);

    	mln.rnnSetPreviousState(0, currStateL0);
    	assertTrue(mln.rnnGetPreviousState(0).size()==2);
    	mln.rnnSetPreviousState(1, currStateL1);
    	assertTrue(mln.rnnGetPreviousState(1).size()==2);
    }
    
    @Test
    public void testRnnTimeStepGravesLSTM(){
    	Nd4j.getRandom().setSeed(12345);
    	int timeSeriesLength = 12;

    	//4 layer network: 2 GravesLSTM + DenseLayer + RnnOutputLayer. Hence also tests preprocessors.
    	MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    		.seed(12345)
    		.list(4)
    		.layer(0,new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder()
    			.nIn(5).nOut(7).activation("tanh").weightInit(WeightInit.DISTRIBUTION)
    			.dist(new NormalDistribution(0,0.5)).build())
    		.layer(1,new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder()
    			.nIn(7).nOut(8).activation("tanh").weightInit(WeightInit.DISTRIBUTION)
    			.dist(new NormalDistribution(0,0.5)).build())
			.layer(2, new DenseLayer.Builder().nIn(8).nOut(9).activation("tanh")
					.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,0.5)).build())
    		.layer(3, new RnnOutputLayer.Builder(LossFunction.MCXENT).weightInit(WeightInit.DISTRIBUTION)
    			.nIn(9).nOut(4).activation("softmax").weightInit(WeightInit.DISTRIBUTION)
    			.dist(new NormalDistribution(0,0.5)).build())
    		.inputPreProcessor(2, new RnnToFeedForwardPreProcessor())
    		.inputPreProcessor(3, new FeedForwardToRnnPreProcessor())
    		.build();
    	MultiLayerNetwork mln = new MultiLayerNetwork(conf);

    	INDArray input = Nd4j.rand(new int[]{3,5,timeSeriesLength});

    	List<INDArray> allOutputActivations = mln.feedForward(input, true);
    	INDArray fullOutL0 = allOutputActivations.get(1);
    	INDArray fullOutL1 = allOutputActivations.get(2);
    	INDArray fullOutL3 = allOutputActivations.get(4);

    	int[] inputLengths = {1,2,3,4,6,12};

    	//Do steps of length 1, then of length 2, ..., 12
    	//Should get the same result regardless of step size; should be identical to standard forward pass
    	for( int i=0; i<inputLengths.length; i++ ){
    		int inLength = inputLengths[i];
    		int nSteps = timeSeriesLength / inLength;	//each of length inLength

    		mln.rnnClearPreviousState();
    		mln.setInputMiniBatchSize(1);	//Reset; should be set by rnnTimeStep method

    		for( int j=0; j<nSteps; j++ ){
    			int startTimeRange = j*inLength;
    			int endTimeRange = startTimeRange + inLength;

    			INDArray inputSubset;
    			if(inLength==1){	//Workaround to nd4j bug
    				int[] sizes = new int[]{input.size(0),input.size(1),1};
    				inputSubset = Nd4j.create(sizes);
    				inputSubset.tensorAlongDimension(0,1,0).assign(input.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(startTimeRange)));
    			} else {
    				inputSubset = input.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(startTimeRange, endTimeRange));
    			}
    			if(inLength>1) assertTrue(inputSubset.size(2)==inLength);

    			INDArray out = mln.rnnTimeStep(inputSubset);

    			INDArray expOutSubset;
    			if(inLength==1){
    				int[] sizes = new int[]{fullOutL3.size(0),fullOutL3.size(1),1};
    				expOutSubset = Nd4j.create(sizes);
    				expOutSubset.tensorAlongDimension(0,1,0).assign(fullOutL3.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(startTimeRange)));
    			}else{
    				expOutSubset = fullOutL3.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.interval(startTimeRange, endTimeRange));
    			}

    			assertTrue(out.equals(expOutSubset));

    			Map<String,INDArray> currL0State = mln.rnnGetPreviousState(0);
    			Map<String,INDArray> currL1State = mln.rnnGetPreviousState(1);

    			INDArray lastActL0 = currL0State.get(GravesLSTM.STATE_KEY_PREV_ACTIVATION);
    			INDArray lastActL1 = currL1State.get(GravesLSTM.STATE_KEY_PREV_ACTIVATION);

    			INDArray expLastActL0 = fullOutL0.tensorAlongDimension(endTimeRange-1, 1,0);
    			INDArray expLastActL1 = fullOutL1.tensorAlongDimension(endTimeRange-1, 1,0);

    			assertTrue(lastActL0.equals(expLastActL0));
    			assertTrue(lastActL1.equals(expLastActL1));
    		}
    	}
    }

    @Test
    public void testRnnTimeStep2dInput(){
    	Nd4j.getRandom().setSeed(12345);
    	int timeSeriesLength = 6;

    	MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    		.list(3)
    		.layer(0,new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder()
    			.nIn(5).nOut(7).activation("tanh").weightInit(WeightInit.DISTRIBUTION)
    			.dist(new NormalDistribution(0,0.5)).build())
    		.layer(1,new org.deeplearning4j.nn.conf.layers.GravesLSTM.Builder()
    			.nIn(7).nOut(8).activation("tanh").weightInit(WeightInit.DISTRIBUTION)
    			.dist(new NormalDistribution(0,0.5)).build())
    		.layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).weightInit(WeightInit.DISTRIBUTION)
    			.nIn(8).nOut(4).activation("softmax").weightInit(WeightInit.DISTRIBUTION)
    			.dist(new NormalDistribution(0,0.5)).build())
    		.build();
    	MultiLayerNetwork mln = new MultiLayerNetwork(conf);
    	mln.init();

    	INDArray input3d = Nd4j.rand(new int[]{3,5,timeSeriesLength});
    	INDArray out3d = mln.rnnTimeStep(input3d);
    	assertArrayEquals(out3d.shape(),new int[]{3,4,timeSeriesLength});

    	mln.rnnClearPreviousState();
    	for( int i=0; i<timeSeriesLength; i++ ){
    		INDArray input2d = input3d.tensorAlongDimension(i,1,0);
    		INDArray out2d = mln.rnnTimeStep(input2d);

    		assertArrayEquals(out2d.shape(),new int[]{3,4});

    		INDArray expOut2d = out3d.tensorAlongDimension(i, 1,0);
    		assertTrue(out2d.equals(expOut2d));
    	}

    	//Check same but for input of size [3,5,1]. Expect [3,4,1] out
    	mln.rnnClearPreviousState();
    	for( int i=0; i<timeSeriesLength; i++ ){
    		INDArray temp = Nd4j.create(new int[]{3,5,1});
    		temp.tensorAlongDimension(0, 1,0).assign(input3d.tensorAlongDimension(i, 1,0));
    		INDArray out3dSlice = mln.rnnTimeStep(temp);
    		assertArrayEquals(out3dSlice.shape(),new int[]{3,4,1});

    		assertTrue(out3dSlice.tensorAlongDimension(0, 1,0).equals(out3d.tensorAlongDimension(i, 1,0)));
    	}
    }
}
