package org.deeplearning4j.optimize.solver;

import static org.junit.Assert.*;

import java.util.Collection;
import java.util.Map;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.override.ConfOverride;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.solvers.ConjugateGradient;
import org.deeplearning4j.optimize.solvers.LineGradientDescent;
import org.deeplearning4j.optimize.solvers.StochasticGradientDescent;
import org.deeplearning4j.optimize.solvers.LBFGS;
import org.deeplearning4j.optimize.stepfunctions.DefaultStepFunction;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class TestOptimizers {
	
	@Test
	public void testOptimizersBasicMLPBackprop(){
		//Basic tests of the 'does it throw an exception' variety.
		
		DataSetIterator iter = new IrisDataSetIterator(5,50);
		
		for( OptimizationAlgorithm oa : OptimizationAlgorithm.values() ){
			MultiLayerNetwork network = new MultiLayerNetwork(getMLPConfigIris(oa));
			network.init();
			
			iter.reset();
			network.fit(iter);
		}
	}
	
	private static MultiLayerConfiguration getMLPConfigIris( OptimizationAlgorithm oa ){
		MultiLayerConfiguration c = new NeuralNetConfiguration.Builder()
		.nIn(4).nOut(3)
		.weightInit(WeightInit.DISTRIBUTION)
		.dist(new NormalDistribution(0, 0.1))

		.activationFunction("sigmoid")
		.lossFunction(LossFunction.MCXENT)
		.optimizationAlgo(oa)
		.iterations(1)
		.batchSize(5)
		.constrainGradientToUnitNorm(false)
		.corruptionLevel(0.0)
		.layer(new RBM())
		.learningRate(0.1).useAdaGrad(false)
		.regularization(true)
		.l2(0.01)
		.applySparsity(false).sparsity(0.0)
		.seed(12345L)
		.list(4).hiddenLayerSizes(8,10,5)
		.backward(true).pretrain(false)
		.useDropConnect(false)

		.override(3, new ConfOverride() {
			@Override
			public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
				builder.activationFunction("softmax");
				builder.layer(new OutputLayer());
				builder.weightInit(WeightInit.DISTRIBUTION);
				builder.dist(new NormalDistribution(0, 0.1));
			}
		}).build();

		return c;
	}
	
	@Test
	public void testSphereFnOptStochGradDescent(){
		testSphereFnOptHelper(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT,-1,2);
		testSphereFnOptHelper(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT,-1,10);
		testSphereFnOptHelper(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT,-1,100);
	}
	
	@Test
	public void testSphereFnOptLineGradDescent(){
		int[] numLineSearchIter = {1,2,5,10};
		for( int n : numLineSearchIter )
			testSphereFnOptHelper(OptimizationAlgorithm.LINE_GRADIENT_DESCENT,n,2);
		
		for( int n : numLineSearchIter )
			testSphereFnOptHelper(OptimizationAlgorithm.LINE_GRADIENT_DESCENT,n,10);

		for( int n : numLineSearchIter )
			testSphereFnOptHelper(OptimizationAlgorithm.LINE_GRADIENT_DESCENT,n,100);
	}
	
	@Test
	public void testSphereFnOptCG(){
		int[] numLineSearchIter = {1,2,5,10};
		for( int n : numLineSearchIter )
			testSphereFnOptHelper(OptimizationAlgorithm.CONJUGATE_GRADIENT,n,2);
		
		for( int n : numLineSearchIter )
			testSphereFnOptHelper(OptimizationAlgorithm.CONJUGATE_GRADIENT,n,10);
		
		for( int n : numLineSearchIter )
			testSphereFnOptHelper(OptimizationAlgorithm.CONJUGATE_GRADIENT,n,100);
	}
	
	@Test
	public void testSphereFnOptLBFGS(){
		int[] numLineSearchIter = {1,2,5,10};
		for( int n : numLineSearchIter )
			testSphereFnOptHelper(OptimizationAlgorithm.LBFGS,n,2);
		
		for( int n : numLineSearchIter )
			testSphereFnOptHelper(OptimizationAlgorithm.LBFGS,n,10);
		
		for( int n : numLineSearchIter )
			testSphereFnOptHelper(OptimizationAlgorithm.LBFGS,n,100);
	}
	
	//For debugging.
	private static final boolean PRINT_OPT_RESULTS = true;
	public void testSphereFnOptHelper( OptimizationAlgorithm oa, int numLineSearchIter, int nDimensions ){
		
		if( PRINT_OPT_RESULTS ) System.out.println("---------\n Alg=" + oa
				+ ", nIter=" + numLineSearchIter + ", nDimensions=" + nDimensions );
		
		NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
		.numLineSearchIterations(numLineSearchIter)
		.iterations(1000)
		.learningRate(0.01)
		.layer(new RBM()).batchSize(1).build();
		conf.addVariable("x");	//Normally done by ParamInitializers, but obviously that isn't done here 
		
		Random rng = new DefaultRandom(12345L);
		org.nd4j.linalg.api.rng.distribution.Distribution dist
			= new org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution(rng,-10, 10);
		Model m = new SphereFunctionModel(nDimensions,dist,conf);
		
		double scoreBefore = m.score();
		assertTrue(!Double.isNaN(scoreBefore) && !Double.isInfinite(scoreBefore));
		if( PRINT_OPT_RESULTS ){
			System.out.println("Before:");
			System.out.println(scoreBefore);
			System.out.println(m.params());
		}
		
		ConvexOptimizer opt;
		switch(oa){
		case STOCHASTIC_GRADIENT_DESCENT:
			opt = new StochasticGradientDescent(conf,new DefaultStepFunction(),null,m);
			break;
		case LINE_GRADIENT_DESCENT:
			opt = new LineGradientDescent(conf,new DefaultStepFunction(),null,m);
			break;
		case CONJUGATE_GRADIENT:
			opt = new ConjugateGradient(conf,new DefaultStepFunction(),null,m);
			break;
		case LBFGS:
			opt = new LBFGS(conf,new DefaultStepFunction(),null,m);
			break;
		default:
			fail("Not supported: " + oa);	//Hessian free is NN-specific.
			opt = null;
			break;
		}
		
		opt.setupSearchState(m.gradientAndScore());
		opt.optimize();
		
		double scoreAfter = m.score();
		assertTrue(!Double.isNaN(scoreAfter) && !Double.isInfinite(scoreAfter));
		if( PRINT_OPT_RESULTS ){
			System.out.println("After:");
			System.out.println(scoreAfter);
			System.out.println(m.params());
		}
		
		//Expected behaviour after optimization:
		//(a) score is better (lower) after optimization.
		//(b) Parameters are closer to minimum after optimization (TODO)
		assertTrue("Score did not improve after optimization (b="+scoreBefore+",a="+scoreAfter+")",scoreAfter < scoreBefore);
		
	}
	
	
	
	
	/** A non-NN optimization problem. Optimization function (cost function) is 
	 * \sum_i x_i^2. Has minimum of 0.0 at x_i=0 for all x_i
	 * See: https://en.wikipedia.org/wiki/Test_functions_for_optimization
	 */
	private static class SphereFunctionModel implements Model, Layer {
		private static final long serialVersionUID = 239156313657395826L;
		private INDArray parameters;
		private final NeuralNetConfiguration conf;
		
		/**@param parameterInit Initial parameters. Also determines dimensionality of problem. Should be row vector.
		 */
		private SphereFunctionModel( INDArray parameterInit, NeuralNetConfiguration conf ){
			this.parameters = parameterInit.dup();
			this.conf = conf;
		}
		
		private SphereFunctionModel( int nParams, org.nd4j.linalg.api.rng.distribution.Distribution distribution,
				NeuralNetConfiguration conf ){
			this.parameters = distribution.sample(new int[]{1,nParams});
			this.conf = conf;
		}

		@Override
		public void fit() { throw new UnsupportedOperationException(); }

		@Override
		public void update(INDArray gradient, String paramType) {
			if(!"x".equals(paramType)) throw new UnsupportedOperationException();
			parameters.subi(gradient);
		}

		@Override
		public double score() {
			return Nd4j.getBlasWrapper().dot(parameters, parameters);	//sum_i x_i^2
		}

		@Override
		public void setScore() { }

		@Override
		public void accumulateScore(double accum) { throw new UnsupportedOperationException(); }

		@Override
		public INDArray transform(INDArray data) { throw new UnsupportedOperationException(); }

		@Override
		public INDArray params() {return parameters; }

		@Override
		public int numParams() { return parameters.length(); }

		@Override
		public void setParams(INDArray params) { this.parameters = params; }

		@Override
		public void fit(INDArray data) { throw new UnsupportedOperationException(); }

		@Override
		public void iterate(INDArray input) { throw new UnsupportedOperationException(); }

		@Override
		public Gradient gradient() {
			// Gradients: d(x^2)/dx = 2x
			INDArray gradient = parameters.mul(2);
			Gradient g = new DefaultGradient();
			g.gradientForVariable().put("x", gradient);
			return g;
		}

		@Override
		public Pair<Gradient, Double> gradientAndScore() {
			return new Pair<Gradient,Double>(gradient(),score());
		}

		@Override
		public int batchSize() { return 1; }

		@Override
		public NeuralNetConfiguration conf() { return conf; }

		@Override
		public void setConf(NeuralNetConfiguration conf) { throw new UnsupportedOperationException(); }

		@Override
		public INDArray input() {
			//Work-around for BaseUpdater.postApply(): Uses Layer.input().size(0)
			//in order to get mini-batch size. i.e., divide by 1 here.
			return Nd4j.zeros(1);
		}

		@Override
		public void validateInput() { }

		@Override
		public ConvexOptimizer getOptimizer() { throw new UnsupportedOperationException(); }

		@Override
		public INDArray getParam(String param) { return parameters; }

		@Override
		public void initParams() { throw new UnsupportedOperationException(); }

		@Override
		public Map<String, INDArray> paramTable() { throw new UnsupportedOperationException(); }

		@Override
		public void setParamTable(Map<String, INDArray> paramTable) { throw new UnsupportedOperationException(); }

		@Override
		public void setParam(String key, INDArray val) { throw new UnsupportedOperationException(); }

		@Override
		public void clear() { throw new UnsupportedOperationException(); }

		@Override
		public Type type() { throw new UnsupportedOperationException(); }

		@Override
		public Gradient error(INDArray input) { throw new UnsupportedOperationException(); }

		@Override
		public INDArray derivativeActivation(INDArray input) { throw new UnsupportedOperationException(); }

		@Override
		public Gradient calcGradient(Gradient layerError, INDArray indArray) { throw new UnsupportedOperationException(); }

		@Override
		public Gradient errorSignal(Gradient error, INDArray input){ throw new UnsupportedOperationException(); }

		@Override
		public Gradient backwardGradient(INDArray z, Layer nextLayer,
				Gradient nextGradient, INDArray activation) { throw new UnsupportedOperationException(); }

		@Override
		public void merge(Layer layer, int batchSize) { throw new UnsupportedOperationException(); }

		@Override
		public INDArray activationMean() { throw new UnsupportedOperationException(); }

		@Override
		public INDArray preOutput(INDArray x) { throw new UnsupportedOperationException(); }

		@Override
		public INDArray activate() { throw new UnsupportedOperationException(); }

		@Override
		public INDArray activate(INDArray input) { throw new UnsupportedOperationException(); }

		@Override
		public Layer transpose() { throw new UnsupportedOperationException(); }

		@Override
		public Layer clone() { throw new UnsupportedOperationException(); }

		@Override
		public Pair<Gradient, Gradient> backWard(Gradient errors,
				Gradient deltas, INDArray activation, String previousActivation) { throw new UnsupportedOperationException(); }

		@Override
		public Collection<IterationListener> getIterationListeners() { return null; }

		@Override
		public void setIterationListeners(Collection<IterationListener> listeners) { throw new UnsupportedOperationException(); }

		@Override
		public void setIndex(int index) { throw new UnsupportedOperationException(); }

		@Override
		public int getIndex() { throw new UnsupportedOperationException(); }
	}
}
