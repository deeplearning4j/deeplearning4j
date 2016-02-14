package org.deeplearning4j.gradientcheck;

import static org.junit.Assert.*;

import java.util.Random;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**@author Alex Black 14 Aug 2015
 */
public class GradientCheckTests {

    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;

    static {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;
        NDArrayFactory factory = Nd4j.factory();
        factory.setDType(DataBuffer.Type.DOUBLE);
    }

    @Test
    public void testGradientMLP2LayerIrisSimple() {
    	//Parameterized test, testing combinations of:
    	// (a) activation function
    	// (b) Whether to test at random initialization, or after some learning (i.e., 'characteristic mode of operation')
    	// (c) Loss function (with specified output activations)
    	String[] activFns = {"sigmoid","tanh","relu","hardtanh","softplus"};
    	boolean[] characteristic = {false,true};	//If true: run some backprop steps first
    	
    	LossFunction[] lossFunctions = {LossFunction.MCXENT, LossFunction.MSE};
    	String[] outputActivations = {"softmax","tanh"};	//i.e., lossFunctions[i] used with outputActivations[i] here
    	
    	DataSet ds = new IrisDataSetIterator(150,150).next();
        ds.normalizeZeroMeanZeroUnitVariance();
        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();
    	
    	for( String afn : activFns) {
    		for(boolean doLearningFirst : characteristic) {
    			for(int i = 0; i<lossFunctions.length; i++) {
    				LossFunction lf = lossFunctions[i];
    				String outputActivation = outputActivations[i];
    				
			        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			                .regularization(false)
			                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
			                .learningRate(1.0)
			                .seed(12345L)
			                .list(2)
			                .layer(0, new DenseLayer.Builder()
									.nIn(4).nOut(3)
									.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
									.activation(afn)
									.updater(Updater.SGD)
									.build())
							.layer(1, new OutputLayer.Builder(lf)
									.activation(outputActivation)
									.nIn(3).nOut(3)
									.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 1))
									.updater(Updater.SGD)
									.build())
							.pretrain(false).backprop(true)
							.build();
			
			        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
			        mln.init();
			        
			        if(doLearningFirst){
			        	//Run a number of iterations of learning
				        mln.setInput(ds.getFeatures());
				        mln.setLabels(ds.getLabels());
				        mln.computeGradientAndScore();
				        double scoreBefore = mln.score();
				        for( int j=0; j<10; j++ ) mln.fit(ds);
				        mln.computeGradientAndScore();
				        double scoreAfter = mln.score();
				        //Can't test in 'characteristic mode of operation' if not learning
				        String msg = "testGradMLP2LayerIrisSimple() - score did not (sufficiently) decrease during learning - activationFn="
				        		+afn+", lossFn="+lf+", outputActivation="+outputActivation+", doLearningFirst="+doLearningFirst
				        		+" (before="+scoreBefore +", scoreAfter="+scoreAfter+")";
				        assertTrue(msg,scoreAfter < 0.8 *scoreBefore);
			        }
			
			        if( PRINT_RESULTS ){
			        	System.out.println("testGradientMLP2LayerIrisSimpleRandom() - activationFn="+afn+", lossFn="+lf+", outputActivation="+outputActivation
			        		+", doLearningFirst="+doLearningFirst );
			        	for( int j=0; j<mln.getnLayers(); j++ ) System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
			        }
			
			        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
			                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);
			
			        String msg = "testGradMLP2LayerIrisSimple() - activationFn="+afn+", lossFn="+lf+", outputActivation="+outputActivation
			        		+", doLearningFirst="+doLearningFirst;
			        assertTrue(msg,gradOK);
    			}
    		}
    	}
    }
    
    @Test
    public void testGradientMLP2LayerIrisL1L2Simple(){
        //As above (testGradientMLP2LayerIrisSimple()) but with L2, L1, and both L2/L1 applied
        //Need to run gradient through updater, so that L2 can be applied

    	String[] activFns = {"sigmoid","tanh","relu"};
    	boolean[] characteristic = {false,true};	//If true: run some backprop steps first
    	
    	LossFunction[] lossFunctions = {LossFunction.MCXENT, LossFunction.MSE};
    	String[] outputActivations = {"softmax","tanh"};	//i.e., lossFunctions[i] used with outputActivations[i] here
    	
    	DataSet ds = new IrisDataSetIterator(150,150).next();
        ds.normalizeZeroMeanZeroUnitVariance();
        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();
        
        double[] l2vals = {0.4, 0.0, 0.4};
        double[] l1vals = {0.0, 0.5, 0.5};	//i.e., use l2vals[i] with l1vals[i]
    	
    	for( String afn : activFns) {
    		for( boolean doLearningFirst : characteristic) {
    			for( int i = 0; i<lossFunctions.length; i++) {
    				for( int k = 0; k<l2vals.length; k++) {
	    				LossFunction lf = lossFunctions[i];
	    				String outputActivation = outputActivations[i];
	    				double l2 = l2vals[k];
	    				double l1 = l1vals[k];
	    				
				        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				                .regularization(true)
				                .l2(l2).l1(l1)
				                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
				                .seed(12345L)
				                .list(2)
				                .layer(0, new DenseLayer.Builder()
										.nIn(4).nOut(3)
										.weightInit(WeightInit.XAVIER).dist(new NormalDistribution(0, 1))
										.updater(Updater.NONE)
										.activation(afn)
										.build())
								.layer(1, new OutputLayer.Builder(lf)
										.nIn(3).nOut(3)
										.weightInit(WeightInit.XAVIER).dist(new NormalDistribution(0, 1))
										.updater(Updater.NONE)
										.activation(outputActivation)
										.build())
								.pretrain(false).backprop(true)
								.build();
				
				        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
				        mln.init();
				        
				        if(doLearningFirst) {
				        	//Run a number of iterations of learning
					        mln.setInput(ds.getFeatures());
					        mln.setLabels(ds.getLabels());
					        mln.computeGradientAndScore();
					        double scoreBefore = mln.score();
					        for( int j=0; j<10; j++ ) mln.fit(ds);
					        mln.computeGradientAndScore();
					        double scoreAfter = mln.score();
					        //Can't test in 'characteristic mode of operation' if not learning
					        String msg = "testGradMLP2LayerIrisSimple() - score did not (sufficiently) decrease during learning - activationFn="
					        		+afn+", lossFn="+lf+", outputActivation="+outputActivation+", doLearningFirst="+doLearningFirst
					        		+", l2="+l2+", l1="+l1+" (before="+scoreBefore +", scoreAfter="+scoreAfter+")";
					        assertTrue(msg,scoreAfter < 0.8 *scoreBefore);
				        }
				
				        if(PRINT_RESULTS) {
				        	System.out.println("testGradientMLP2LayerIrisSimpleRandom() - activationFn="+afn+", lossFn="+lf+", outputActivation="+outputActivation
				        		+", doLearningFirst="+doLearningFirst +", l2="+l2+", l1="+l1 );
				        	for( int j=0; j<mln.getnLayers(); j++ ) System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
				        }
				
				        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
				                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);
				
				        String msg = "testGradMLP2LayerIrisSimple() - activationFn="+afn+", lossFn="+lf+", outputActivation="+outputActivation
				        		+", doLearningFirst="+doLearningFirst +", l2="+l2+", l1="+l1;
				        assertTrue(msg,gradOK);
    				}
    			}
    		}
    	}
    }

	@Test
	public void testEmbeddingLayerSimple(){
		Random r = new Random(12345);
		int nExamples = 5;
		INDArray input = Nd4j.zeros(nExamples,1);
		INDArray labels = Nd4j.zeros(nExamples,3);
		for( int i=0; i<nExamples; i++ ){
			input.putScalar(i,r.nextInt(4));
			labels.putScalar(new int[]{i,r.nextInt(3)},1.0);
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.regularization(true)
				.l2(0.2).l1(0.1)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.seed(12345L)
				.list(2)
				.layer(0, new EmbeddingLayer.Builder()
						.nIn(4).nOut(3)
						.weightInit(WeightInit.XAVIER).dist(new NormalDistribution(0, 1))
						.updater(Updater.NONE)
						.activation("tanh")
						.build())
				.layer(1, new OutputLayer.Builder(LossFunction.MCXENT)
						.nIn(3).nOut(3)
						.weightInit(WeightInit.XAVIER).dist(new NormalDistribution(0, 1))
						.updater(Updater.NONE)
						.activation("softmax")
						.build())
				.pretrain(false).backprop(true)
				.build();

		MultiLayerNetwork mln = new MultiLayerNetwork(conf);
		mln.init();

		if(PRINT_RESULTS) {
			System.out.println("testEmbeddingLayerSimple");
			for( int j=0; j<mln.getnLayers(); j++ ) System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
		}

		boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
				PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);

		String msg = "testEmbeddingLayerSimple";
		assertTrue(msg,gradOK);
	}
    
    
    @Test
    public void testGRURNNBasicMultiLayer(){
    	//Basic test of GRU RNN
    	Nd4j.getRandom().setSeed(12345L);
    	
    	int timeSeriesLength = 2;
    	int nIn = 2;
    	int gruLayerSize = 2;
    	int nOut = 2;
    	int miniBatchSize = 1;
    	
    	MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	        .regularization(false)
	        .seed(12345L)
	        .list(3)
	        .layer(0, new GRU.Builder().nIn(nIn).nOut(gruLayerSize).activation("tanh")
	        		.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1.0))
	        		.updater(Updater.NONE).build())
	        .layer(1, new GRU.Builder().nIn(gruLayerSize).nOut(gruLayerSize).activation("tanh")
	        		.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1.0))
	        		.updater(Updater.NONE).build())
	        .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax").nIn(gruLayerSize)
	        		.nOut(nOut).weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1.0))
	        		.updater(Updater.NONE).build())
	        .pretrain(false).backprop(true)
	        .build();
    	
    	MultiLayerNetwork mln = new MultiLayerNetwork(conf);
    	mln.init();
    	
    	Random r = new Random(12345L);
    	INDArray input = Nd4j.zeros(miniBatchSize, nIn, timeSeriesLength);
    	for( int i=0; i<miniBatchSize; i++ ){
    		for( int j=0; j<nIn; j++ ){
    			for( int k=0; k<timeSeriesLength; k++ ){
    				input.putScalar(new int[]{i,j,k},r.nextDouble()-0.5);
    			}
    		}
    	}
    	INDArray labels = Nd4j.zeros(miniBatchSize, nOut, timeSeriesLength);
    	for( int i=0; i<miniBatchSize; i++ ){
    		for( int j=0; j<timeSeriesLength; j++ ){
    			int idx = r.nextInt(nOut);
    			labels.putScalar(new int[]{i,idx,j}, 1.0f);
    		}
    	}
    	
    	if(PRINT_RESULTS) {
    		System.out.println("testGRURNNBasicMultiLayer()");
    		for( int j=0; j<mln.getnLayers(); j++ )
                System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
    	}
    	
    	boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);

        assertTrue(gradOK);
    }
    
    @Test
    public void testGradientGRURNNFull(){
    	String[] activFns = {"tanh","relu"};
    	
    	LossFunction[] lossFunctions = {LossFunction.MCXENT, LossFunction.MSE};
    	String[] outputActivations = {"softmax","tanh"};	//i.e., lossFunctions[i] used with outputActivations[i] here
    	
    	int timeSeriesLength = 6;
    	int nIn = 7;
    	int layerSize = 9;
    	int nOut = 4;
    	int miniBatchSize = 8;
    	
    	Random r = new Random(12345L);
    	INDArray input = Nd4j.zeros(miniBatchSize,nIn,timeSeriesLength);
    	for( int i=0; i<miniBatchSize; i++ ){
    		for( int j=0; j<nIn; j++ ){
    			for( int k=0; k<timeSeriesLength; k++ ){
    				input.putScalar(new int[]{i,j,k},r.nextDouble()-0.5);
    			}
    		}
    	}

    	INDArray labels = Nd4j.zeros(miniBatchSize,nOut,timeSeriesLength);
    	for( int i=0; i<miniBatchSize; i++ ){
    		for( int j=0; j<timeSeriesLength; j++ ){
    			int idx = r.nextInt(nOut);
    			labels.putScalar(new int[]{i,idx,j}, 1.0f);
    		}
    	}
        
        double[] l2vals = {0.0, 0.4, 0.0};
        double[] l1vals = {0.0, 0.0, 0.5};	//i.e., use l2vals[i] with l1vals[i]
    	
    	for( String afn : activFns ){
			for( int i=0; i<lossFunctions.length; i++ ){
				for( int k=0; k<l2vals.length; k++ ){
    				LossFunction lf = lossFunctions[i];
    				String outputActivation = outputActivations[i];
    				double l2 = l2vals[k];
    				double l1 = l1vals[k];
    				
			        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			                .regularization(l1>0.0 || l2>0.0)
			                .l2(l2).l1(l1)
			                .seed(12345L)
			                .list(2)
			                .layer(0, new GRU.Builder().nIn(nIn).nOut(layerSize).activation(afn)
			                		.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1))
			                		.updater(Updater.NONE).build())
			                .layer(1, new RnnOutputLayer.Builder(lf).activation(outputActivation).nIn(layerSize).nOut(nOut)
			                		.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1))
			                		.updater(Updater.NONE).build())
			                .pretrain(false).backprop(true)
			                .build();
			
			        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
			        mln.init();
			
			        if(PRINT_RESULTS) {
			        	System.out.println("testGradientGRURNNFull() - activationFn="+afn+", lossFn="+lf+", outputActivation="+outputActivation
			        		+", l2="+l2+", l1="+l1 );
			        	for( int j=0; j<mln.getnLayers(); j++ ) System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
			        }
			
			        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
			                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);
			
			        String msg = "testGradientGRURNNFull() - activationFn="+afn+", lossFn="+lf+", outputActivation="+outputActivation
			        		+", l2="+l2+", l1="+l1;
			        assertTrue(msg,gradOK);
				}
			}
    	}
    }
    
    @Test
    public void testGradientGRUEdgeCases(){
    	//Edge cases: T=1, miniBatchSize=1, both
    	int[] timeSeriesLength = {1,5,1};
    	int[] miniBatchSize = {7,1,1};
    	
    	int nIn = 7;
    	int layerSize = 9;
    	int nOut = 4;
    	
    	for( int i=0; i<timeSeriesLength.length; i++) {
    		
    		Random r = new Random(12345L);
        	INDArray input = Nd4j.zeros(miniBatchSize[i],nIn,timeSeriesLength[i]);
        	for( int m=0; m<miniBatchSize[m]; m++ ){
        		for( int j=0; j<nIn; j++ ){
        			for( int k=0; k<timeSeriesLength[i]; k++ ){
        				input.putScalar(new int[]{m,j,k},r.nextDouble()-0.5);
        			}
        		}
        	}

        	INDArray labels = Nd4j.zeros(miniBatchSize[i],nOut,timeSeriesLength[i]);
        	for( int m=0; m<miniBatchSize[i]; m++ ){
        		for( int j=0; j<timeSeriesLength[i]; j++ ){
        			int idx = r.nextInt(nOut);
        			labels.putScalar(new int[]{m,idx,j}, 1.0f);
        		}
        	}
    		
    		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	            .regularization(false)
	            .seed(12345L)
	            .list(2)
	            .layer(0, new GRU.Builder().nIn(nIn).nOut(layerSize)
	            		.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1)).updater(Updater.NONE).build())
	            .layer(1, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax").nIn(layerSize).nOut(nOut)
	            		.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1)).updater(Updater.NONE).build())
	            .pretrain(false).backprop(true)
	            .build();
    		MultiLayerNetwork mln = new MultiLayerNetwork(conf);
    		mln.init();
    		
    		boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
	                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);
	
	        String msg = "testGradientGRUEdgeCases() - timeSeriesLength="+timeSeriesLength[i]+", miniBatchSize="+miniBatchSize[i];
	        assertTrue(msg,gradOK);
    	}
    }
    
    @Test
    public void testGravesLSTMBasicMultiLayer() {
    	//Basic test of GravesLSTM layer
    	Nd4j.getRandom().setSeed(12345L);
    	
    	int timeSeriesLength = 4;
    	int nIn = 2;
    	int layerSize = 2;
    	int nOut = 2;
    	int miniBatchSize = 5;
    	
    	MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	        .regularization(false)
	        .seed(12345L)
	        .list(3)
	        .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(layerSize).activation("sigmoid")
                .weightInit(WeightInit.XAVIER).dist(new NormalDistribution(0,1.0)).updater(Updater.NONE).build())
	        .layer(1, new GravesLSTM.Builder().nIn(layerSize).nOut(layerSize).activation("sigmoid")
				.weightInit(WeightInit.XAVIER).dist(new NormalDistribution(0,1.0)).updater(Updater.NONE).build())
	        .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax").nIn(layerSize).nOut(nOut)
				.weightInit(WeightInit.XAVIER).dist(new NormalDistribution(0,1.0)).updater(Updater.NONE).build())
	        .pretrain(false).backprop(true)
	        .build();
    	
    	MultiLayerNetwork mln = new MultiLayerNetwork(conf);
    	mln.init();
    	
    	Random r = new Random(12345L);
    	INDArray input = Nd4j.zeros(miniBatchSize,nIn,timeSeriesLength);
    	for( int i = 0; i < miniBatchSize; i++) {
    		for( int j = 0; j < nIn; j++) {
    			for( int k = 0; k < timeSeriesLength; k++) {
    				input.putScalar(new int[]{i,j,k},r.nextDouble() - 0.5);
    			}
    		}
    	}

    	INDArray labels = Nd4j.zeros(miniBatchSize,nOut,timeSeriesLength);
    	for( int i = 0; i < miniBatchSize; i++ ){
    		for(int j = 0; j < timeSeriesLength; j++) {
    			int idx = r.nextInt(nOut);
    			labels.putScalar(new int[]{i,idx,j}, 1.0);
    		}
    	}
    	
    	if(PRINT_RESULTS) {
    		System.out.println("testGravesLSTMBasic()");
    		for( int j = 0; j < mln.getnLayers(); j++ )
				System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
    	}
    	
    	boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);

        assertTrue(gradOK);
    }
    
    @Test
    public void testGradientGravesLSTMFull(){
    	String[] activFns = {"tanh","relu"};
    	
    	LossFunction[] lossFunctions = {LossFunction.MCXENT, LossFunction.MSE};
    	String[] outputActivations = {"softmax","tanh"};	//i.e., lossFunctions[i] used with outputActivations[i] here
    	
    	int timeSeriesLength = 8;
    	int nIn = 7;
    	int layerSize = 9;
    	int nOut = 4;
    	int miniBatchSize = 6;
    	
    	Random r = new Random(12345L);
    	INDArray input = Nd4j.zeros(miniBatchSize,nIn,timeSeriesLength);
    	for( int i=0; i<miniBatchSize; i++ ){
    		for( int j=0; j<nIn; j++ ){
    			for( int k=0; k<timeSeriesLength; k++ ){
    				input.putScalar(new int[]{i,j,k},r.nextDouble() - 0.5);
    			}
    		}
    	}
    	
    	INDArray labels = Nd4j.zeros(miniBatchSize,nOut,timeSeriesLength);
    	for( int i=0; i<miniBatchSize; i++ ){
    		for( int j=0; j<timeSeriesLength; j++ ){
    			int idx = r.nextInt(nOut);
    			labels.putScalar(new int[]{i,idx,j}, 1.0f);
    		}
    	}
    	
        
        double[] l2vals = {0.0, 0.4, 0.0};
        double[] l1vals = {0.0, 0.0, 0.5};	//i.e., use l2vals[i] with l1vals[i]
    	
    	for(String afn : activFns) {
			for( int i = 0; i<lossFunctions.length; i++ ){
				for( int k = 0; k<l2vals.length; k++ ){
    				LossFunction lf = lossFunctions[i];
    				String outputActivation = outputActivations[i];
    				double l2 = l2vals[k];
    				double l1 = l1vals[k];
    				
			        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			                .regularization(l1>0.0 || l2>0.0)
			                .l2(l2).l1(l1)
			                .seed(12345L)
			                .list(2)
			                .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(layerSize)
			                		.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1))
			                		.activation(afn).updater(Updater.NONE).build())
			                .layer(1, new RnnOutputLayer.Builder(lf).activation(outputActivation).nIn(layerSize).nOut(nOut)
			                		.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1))
			                		.updater(Updater.NONE).build())
			                .pretrain(false).backprop(true)
			                .build();
			
			        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
			        mln.init();

			        if( PRINT_RESULTS ){
			        	System.out.println("testGradientGravesLSTMFull() - activationFn="+afn+", lossFn="+lf+", outputActivation="+outputActivation
			        		+", l2="+l2+", l1="+l1 );
			        	for( int j=0; j < mln.getnLayers(); j++ )
                            System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
			        }
			
			        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
			                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);
			
			        String msg = "testGradientGravesLSTMFull() - activationFn="+afn+", lossFn="+lf+", outputActivation="+outputActivation
			        		+", l2="+l2+", l1="+l1;
			        assertTrue(msg,gradOK);
				}
			}
    	}
    }


    
    @Test
    public void testGradientGravesLSTMEdgeCases(){
    	//Edge cases: T=1, miniBatchSize=1, both
    	int[] timeSeriesLength = {1,5,1};
    	int[] miniBatchSize = {7,1,1};
    	
    	int nIn = 7;
    	int layerSize = 9;
    	int nOut = 4;
    	
    	for( int i=0; i<timeSeriesLength.length; i++ ){
    		
    		Random r = new Random(12345L);
        	INDArray input = Nd4j.zeros(miniBatchSize[i],nIn,timeSeriesLength[i]);
        	for( int m=0; m<miniBatchSize[i]; m++ ){
        		for( int j=0; j<nIn; j++ ){
        			for( int k=0; k<timeSeriesLength[i]; k++ ){
        				input.putScalar(new int[]{m,j,k},r.nextDouble() - 0.5);
        			}
        		}
        	}
        	
        	INDArray labels = Nd4j.zeros(miniBatchSize[i],nOut,timeSeriesLength[i]);
        	for( int m=0; m<miniBatchSize[i]; m++){
        		for( int j=0; j<timeSeriesLength[i]; j++ ){
        			int idx = r.nextInt(nOut);
        			labels.putScalar(new int[]{m,idx,j}, 1.0f);
        		}
        	}
    		
    		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	            .regularization(false)
	            .seed(12345L)
	            .list(2)
	            .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(layerSize).weightInit(WeightInit.DISTRIBUTION)
	            		.dist(new NormalDistribution(0,1)).updater(Updater.NONE).build())
	            .layer(1, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax").nIn(layerSize).nOut(nOut)
	            		.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1)).updater(Updater.NONE).build())
	            .pretrain(false).backprop(true)
	            .build();
    		MultiLayerNetwork mln = new MultiLayerNetwork(conf);
    		mln.init();
    		
    		boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
	                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);
	
	        String msg = "testGradientGravesLSTMEdgeCases() - timeSeriesLength="+timeSeriesLength[i]+", miniBatchSize="+miniBatchSize[i];
	        assertTrue(msg,gradOK);
    	}
    }

	@Test
	public void testGradientGravesBidirectionalLSTMFull(){
		String[] activFns = {"tanh","relu"};

		LossFunction[] lossFunctions = {LossFunction.MCXENT, LossFunction.MSE};
		String[] outputActivations = {"softmax","tanh"};	//i.e., lossFunctions[i] used with outputActivations[i] here

		int timeSeriesLength = 4;
		int nIn = 1;
		int layerSize = 1;
		int nOut = 2;
		int miniBatchSize = 3;

		Random r = new Random(12345L);
		INDArray input = Nd4j.zeros(miniBatchSize,nIn,timeSeriesLength);
		for( int i=0; i<miniBatchSize; i++ ){
			for( int j=0; j<nIn; j++ ){
				for( int k=0; k<timeSeriesLength; k++ ){
					input.putScalar(new int[]{i,j,k},r.nextDouble() - 0.5);
				}
			}
		}

		INDArray labels = Nd4j.zeros(miniBatchSize,nOut,timeSeriesLength);
		for( int i=0; i<miniBatchSize; i++ ){
			for( int j=0; j<timeSeriesLength; j++ ){
				int idx = r.nextInt(nOut);
				labels.putScalar(new int[]{i,idx,j}, 1.0f);
			}
		}


		double[] l2vals = {0.0, 0.4, 0.0};
		double[] l1vals = {0.0, 0.0, 0.5};	//i.e., use l2vals[i] with l1vals[i]

		for(String afn : activFns) {
			for( int i = 0; i<lossFunctions.length; i++ ){
				for( int k = 0; k<l2vals.length; k++ ){
					LossFunction lf = lossFunctions[i];
					String outputActivation = outputActivations[i];
					double l2 = l2vals[k];
					double l1 = l1vals[k];

					MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
							.regularization(l1>0.0 || l2>0.0)
							.l2(l2).l1(l1)
							.seed(12345L)
							.list(2)
							.layer(0, new GravesBidirectionalLSTM.Builder().nIn(nIn).nOut(layerSize)
									.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1))
									.activation(afn).updater(Updater.NONE).build())
							.layer(1, new RnnOutputLayer.Builder(lf).activation(outputActivation).nIn(layerSize).nOut(nOut)
									.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1))
									.updater(Updater.NONE).build())
							.pretrain(false).backprop(true)
							.build();



					MultiLayerNetwork mln = new MultiLayerNetwork(conf);

					mln.init();

					if( PRINT_RESULTS ){
						System.out.println("testGradientGravesBidirectionalLSTMFull() - activationFn="+afn+", lossFn="+lf+", outputActivation="+outputActivation
								+", l2="+l2+", l1="+l1 );
						for( int j=0; j < mln.getnLayers(); j++ )
							System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
					}

					boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
							PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);

					String msg = "testGradientGravesLSTMFull() - activationFn="+afn+", lossFn="+lf+", outputActivation="+outputActivation
							+", l2="+l2+", l1="+l1;
					assertTrue(msg,gradOK);
				}
			}
		}
	}

	@Test
	public void testGradientGravesBidirectionalLSTMEdgeCases(){
		//Edge cases: T=1, miniBatchSize=1, both
		int[] timeSeriesLength = {1,5,1};
		int[] miniBatchSize = {7,1,1};

		int nIn = 7;
		int layerSize = 9;
		int nOut = 4;

		for( int i=0; i<timeSeriesLength.length; i++ ){

			Random r = new Random(12345L);
			INDArray input = Nd4j.zeros(miniBatchSize[i],nIn,timeSeriesLength[i]);
			for( int m=0; m<miniBatchSize[i]; m++ ){
				for( int j=0; j<nIn; j++ ){
					for( int k=0; k<timeSeriesLength[i]; k++ ){
						input.putScalar(new int[]{m,j,k},r.nextDouble() - 0.5);
					}
				}
			}

			INDArray labels = Nd4j.zeros(miniBatchSize[i],nOut,timeSeriesLength[i]);
			for( int m=0; m<miniBatchSize[i]; m++){
				for( int j=0; j<timeSeriesLength[i]; j++ ){
					int idx = r.nextInt(nOut);
					labels.putScalar(new int[]{m,idx,j}, 1.0f);
				}
			}

			MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
					.regularization(false)
					.seed(12345L)
					.list(2)
					.layer(0, new GravesBidirectionalLSTM.Builder().nIn(nIn).nOut(layerSize).weightInit(WeightInit.DISTRIBUTION)
							.dist(new NormalDistribution(0,1)).updater(Updater.NONE).build())
					.layer(1, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax").nIn(layerSize).nOut(nOut)
							.weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,1)).updater(Updater.NONE).build())
					.pretrain(false).backprop(true)
					.build();
			MultiLayerNetwork mln = new MultiLayerNetwork(conf);
			mln.init();

			boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
					PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);

			String msg = "testGradientGravesLSTMEdgeCases() - timeSeriesLength="+timeSeriesLength[i]+", miniBatchSize="+miniBatchSize[i];
			assertTrue(msg,gradOK);
		}
	}

	@Test
	public void testGradientCnnFfRnn(){
		//Test gradients with CNN -> FF -> LSTM -> RnnOutputLayer
		//time series input/output (i.e., video classification or similar)

		int nChannelsIn = 3;
		int inputSize = 10*10*nChannelsIn;	//10px x 10px x 3 channels
		int miniBatchSize = 4;
		int timeSeriesLength = 10;
		int nClasses = 3;

		//Generate
		Nd4j.getRandom().setSeed(12345);
		INDArray input = Nd4j.rand(new int[]{miniBatchSize,inputSize,timeSeriesLength});
		INDArray labels = Nd4j.zeros(miniBatchSize, nClasses, timeSeriesLength);
		Random r = new Random(12345);
		for( int i = 0; i < miniBatchSize; i++ ){
			for(int j = 0; j < timeSeriesLength; j++) {
				int idx = r.nextInt(nClasses);
				labels.putScalar(new int[]{i,idx,j}, 1.0);
			}
		}


		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(12345)
				.list(5)
				.layer(0, new ConvolutionLayer.Builder(5, 5)
						.nIn(3)
						.nOut(5)
						.stride(1, 1)
						.activation("relu")
						.weightInit(WeightInit.XAVIER)
						.updater(Updater.NONE)
						.build())	//Out: (10-5)/1+1 = 6 -> 6x6x5
				.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(1, 1)
						.build())	//Out: (6-2)/1+1 = 5 -> 5x5x5
				.layer(2, new DenseLayer.Builder()
						.nIn(5 * 5 * 5)
						.nOut(4)
						.updater(Updater.NONE)
						.weightInit(WeightInit.XAVIER)
						.activation("relu")
						.build())
				.layer(3, new GravesLSTM.Builder()
						.nIn(4)
						.nOut(3)
						.updater(Updater.NONE)
						.weightInit(WeightInit.XAVIER)
						.activation("tanh")
						.build())
				.layer(4, new RnnOutputLayer.Builder()
						.lossFunction(LossFunction.MCXENT)
						.nIn(3)
						.nOut(nClasses)
						.activation("softmax")
						.updater(Updater.NONE)
						.build())
				.cnnInputSize(10, 10, 3)
				.pretrain(false).backprop(true)
				.build();

		//Here: ConvolutionLayerSetup in config builder doesn't know that we are expecting time series input, not standard FF input -> override it here
		conf.getInputPreProcessors().put(0,new RnnToCnnPreProcessor(10, 10, 3));

		MultiLayerNetwork mln = new MultiLayerNetwork(conf);
		mln.init();

		System.out.println("Params per layer:");
		for( int i = 0; i<mln.getnLayers(); i++ ){
			System.out.println("layer " + i + "\t" + mln.getLayer(i).numParams());
		}

		boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
				PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);
		assertTrue(gradOK);
	}


    @Test
    @Ignore
    public void testRbm() {
        //As above (testGradientMLP2LayerIrisSimple()) but with L2, L1, and both L2/L1 applied
        //Need to run gradient through updater, so that L2 can be applied

        String[] activFns = {"sigmoid","relu"};
        boolean[] characteristic = {false,true};	//If true: run some backprop steps first

        LossFunction[] lossFunctions = {LossFunction.MCXENT, LossFunction.MSE};
        String[] outputActivations = {"softmax","tanh"};	//i.e., lossFunctions[i] used with outputActivations[i] here

        DataSet ds = new IrisDataSetIterator(150,150).next();
        ds.scaleMinAndMax(0,1);
        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();

        double[] l2vals = {0.4, 0.0, 0.4};
        double[] l1vals = {0.0, 0.5, 0.5};	//i.e., use l2vals[i] with l1vals[i]

        for( String afn : activFns) {
            for( boolean doLearningFirst : characteristic) {
                for( int i = 0; i < lossFunctions.length; i++) {
                    for( int k = 0; k < l2vals.length; k++) {
                        LossFunction lf = lossFunctions[i];
                        String outputActivation = outputActivations[i];
                        double l2 = l2vals[k];
                        double l1 = l1vals[k];

                        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                .regularization(true)
                                .l2(l2).l1(l1)
                                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                                .seed(12345L)
                                .list(2)
                                .layer(0, new RBM.Builder(RBM.HiddenUnit.BINARY, RBM.VisibleUnit.BINARY)
                                        .nIn(4).nOut(3)
                                        .weightInit(WeightInit.XAVIER)
                                        .updater(Updater.SGD)
                                        .activation(afn)
                                        .build())
                                .layer(1, new OutputLayer.Builder(lf)
                                        .nIn(3).nOut(3)
                                        .weightInit(WeightInit.XAVIER)
                                        .updater(Updater.SGD)
                                        .activation(outputActivation)
                                        .build())
                                .pretrain(true).backprop(true)
                                .build();

                        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                        mln.init();

                        if(doLearningFirst) {
                            //Run a number of iterations of learning
                            mln.setInput(ds.getFeatures());
                            mln.setLabels(ds.getLabels());
                            mln.computeGradientAndScore();
                            double scoreBefore = mln.score();
                            for( int j = 0; j < 10; j++)
                                mln.fit(ds);
                            mln.computeGradientAndScore();
                            double scoreAfter = mln.score();
                            //Can't test in 'characteristic mode of operation' if not learning
                            String msg = "testGradMLP2LayerIrisSimple() - score did not (sufficiently) decrease during learning - activationFn="
                                    +afn+", lossFn=" + lf + ", outputActivation=" + outputActivation + ", doLearningFirst=" + doLearningFirst
                                    +", l2="+l2+", l1="+l1+" (before="+scoreBefore +", scoreAfter=" + scoreAfter+")";
                            assertTrue(msg,scoreAfter < 0.8 *scoreBefore);
                        }

                        if(PRINT_RESULTS) {
                            System.out.println("testGradientMLP2LayerIrisSimpleRandom() - activationFn="+afn+", lossFn="+lf+", outputActivation="+outputActivation
                                    +", doLearningFirst=" + doLearningFirst +", l2=" + l2+", l1=" + l1);
                            for( int j = 0; j<mln.getnLayers(); j++)
                                System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
                        }

                        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);

                        String msg = "testGradMLP2LayerIrisSimple() - activationFn="+afn+", lossFn="+lf+", outputActivation="+outputActivation
                                +", doLearningFirst=" + doLearningFirst +", l2=" + l2 + ", l1=" + l1;
                        assertTrue(msg,gradOK);
                    }
                }
            }
        }
    }

    @Test
    public void testAutoEncoder() {
        //As above (testGradientMLP2LayerIrisSimple()) but with L2, L1, and both L2/L1 applied
        //Need to run gradient through updater, so that L2 can be applied

        String[] activFns = {"sigmoid","tanh","relu"};
        boolean[] characteristic = {false,true};	//If true: run some backprop steps first

        LossFunction[] lossFunctions = {LossFunction.MCXENT, LossFunction.MSE};
        String[] outputActivations = {"softmax","tanh"};	//i.e., lossFunctions[i] used with outputActivations[i] here

        DataSet ds = new IrisDataSetIterator(150,150).next();
        ds.normalizeZeroMeanZeroUnitVariance();
        INDArray input = ds.getFeatureMatrix();
        INDArray labels = ds.getLabels();

        double[] l2vals = {0.4, 0.0, 0.4};
        double[] l1vals = {0.0, 0.5, 0.5};	//i.e., use l2vals[i] with l1vals[i]

        for( String afn : activFns) {
            for( boolean doLearningFirst : characteristic) {
                for( int i = 0; i < lossFunctions.length; i++) {
                    for( int k = 0; k < l2vals.length; k++) {
                        LossFunction lf = lossFunctions[i];
                        String outputActivation = outputActivations[i];
                        double l2 = l2vals[k];
                        double l1 = l1vals[k];

                        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                .regularization(true)
                                .l2(l2).l1(l1)
                                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                                .seed(12345L)
                                .list(2)
                                .layer(0, new AutoEncoder.Builder()
                                        .nIn(4).nOut(3)
                                        .weightInit(WeightInit.XAVIER).dist(new NormalDistribution(0, 1))
                                        .updater(Updater.NONE)
                                        .activation(afn)
                                        .build())
                                .layer(1, new OutputLayer.Builder(lf)
                                        .nIn(3).nOut(3)
                                        .weightInit(WeightInit.XAVIER).dist(new NormalDistribution(0, 1))
                                        .updater(Updater.NONE)
                                        .activation(outputActivation)
                                        .build())
                                .pretrain(true).backprop(true)
                                .build();

                        MultiLayerNetwork mln = new MultiLayerNetwork(conf);
                        mln.init();

                        if(doLearningFirst) {
                            //Run a number of iterations of learning
                            mln.setInput(ds.getFeatures());
                            mln.setLabels(ds.getLabels());
                            mln.computeGradientAndScore();
                            double scoreBefore = mln.score();
                            for( int j = 0; j < 10; j++)
                                mln.fit(ds);
                            mln.computeGradientAndScore();
                            double scoreAfter = mln.score();
                            //Can't test in 'characteristic mode of operation' if not learning
                            String msg = "testGradMLP2LayerIrisSimple() - score did not (sufficiently) decrease during learning - activationFn="
                                    +afn+", lossFn=" + lf + ", outputActivation=" + outputActivation + ", doLearningFirst=" + doLearningFirst
                                    +", l2="+l2+", l1="+l1+" (before="+scoreBefore +", scoreAfter=" + scoreAfter+")";
                            assertTrue(msg,scoreAfter < 0.8 *scoreBefore);
                        }

                        if(PRINT_RESULTS) {
                            System.out.println("testGradientMLP2LayerIrisSimpleRandom() - activationFn="+afn+", lossFn="+lf+", outputActivation="+outputActivation
                                    +", doLearningFirst=" + doLearningFirst +", l2=" + l2+", l1=" + l1);
                            for( int j = 0; j<mln.getnLayers(); j++)
                                System.out.println("Layer " + j + " # params: " + mln.getLayer(j).numParams());
                        }

                        boolean gradOK = GradientCheckUtil.checkGradients(mln, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                                PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels, true);

                        String msg = "testGradMLP2LayerIrisSimple() - activationFn="+afn+", lossFn="+lf+", outputActivation="+outputActivation
                                +", doLearningFirst="+doLearningFirst +", l2=" + l2 + ", l1="+l1;
                        assertTrue(msg,gradOK);
                    }
                }
            }
        }
    }
}
