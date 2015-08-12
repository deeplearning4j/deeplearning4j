package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.UpdaterCreator;
import org.nd4j.linalg.api.ndarray.INDArray;

/** A utility for numerically checking gradients. <br>
 * Basic idea: compare calculated gradients with those calculated numerically,
 * to check implementation of backpropagation gradient calculation.<br>
 * See:<br>
 * - http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization<br>
 * - https://code.google.com/p/cuda-convnet/wiki/CheckingGradients<br>
 * 
 * Is C is cost function, then dC/dw ~= (C(w+epsilon)-C(w-epsilon)) / (2*epsilon).<br>
 * Method checks gradient calculation for every parameter separately by doing 2 forward pass
 * calculations for each parameter, so can be very time consuming for large networks.
 * 
 * @author Alex Black
 */
public class GradientCheckUtil {
	
	/**Check backprop gradients for a MultiLayerNetwork. 
	 * @param mln MultiLayerNetwork to test. This must be initialized.
	 * @param epsilon. Usually on the order of 1e-4 or so.
	 * @param maxRelError Maximum relative error. Usually around 0.02 to 0.10.
	 * @param print Whether to print full pass/failure details for each parameter gradient
	 * @param exitOnFirstError If true: return upon first failure. If false: continue checking even if
	 *  one parameter gradient has failed. Typically use false for debugging, true for unit tests.
	 * @param input Input array to use for forward pass. May be mini-batch data.
	 * @param labels Labels/targets to use to calculate backprop gradient. May be mini-batch data.
	 * @param useUpdater Whether to put the gradient through Updater.update(...). Necessary for testing things
	 *  like l1 and l2 regularization.
	 * @return true if gradients are passed, false otherwise.
	 */
	public static boolean checkGradients( MultiLayerNetwork mln, double epsilon, double maxRelError,
			boolean print, boolean exitOnFirstError, INDArray input, INDArray labels, boolean useUpdater ){
		//Basic sanity checks on input:
		if( epsilon <= 0.0 || epsilon > 0.1 ) throw new IllegalArgumentException("Invalid epsilon: expect epsilon in range (0,0.1], usually 1e-4 or so");
		if( maxRelError <= 0.0 || maxRelError > 0.25 ) throw new IllegalArgumentException("Invalid maxRelativeError: " + maxRelError ); 
		if( !(mln.getOutputLayer() instanceof OutputLayer) ) throw new IllegalArgumentException("Cannot check backprop gradients without OutputLayer");
		
		mln.setInput(input);
		mln.setLabels(labels);
		
		mln.feedForward();
		mln.computeGradientAndScore();
		Pair<Gradient,Double> gradAndScore = mln.gradientAndScore();
		
		if(useUpdater){
			Updater updater = UpdaterCreator.getUpdater(mln);
			updater.update(mln, gradAndScore.getFirst(), 0);
		}
		
		INDArray gradientToCheck = gradAndScore.getFirst().gradient(mln.conf().getVariables());
		INDArray originalParams = mln.params();
		
		OutputLayer ol = (OutputLayer)mln.getOutputLayer();
		int nParams = mln.numParams();
		
		int totalNFailures = 0;
		double maxError = 0.0;
		for( int i=0; i<nParams; i++ ){
			//(w+epsilon): Do forward pass and score
			INDArray params = originalParams.dup();
			params.putScalar(i, params.getDouble(i)+epsilon);
			mln.setParameters(params);
			mln.feedForward();
			double scorePlus = ol.computeScore();
			
			//(w-epsilon): Do forward pass and score
			params.putScalar(i, params.getDouble(i)-2*epsilon);
			mln.setParameters(params);
			mln.feedForward();
			double scoreMinus = ol.computeScore();
			
			//Calculate numerical parameter gradient:
			double numericalGradient = (scorePlus - scoreMinus) / (2*epsilon);
			if( Double.isNaN(numericalGradient) ) throw new IllegalStateException("Numerical gradient was NaN for parameter " + i + " of " + nParams);
			
			double backpropGradient = gradientToCheck.getDouble(i);
			double relError = Math.abs(numericalGradient-backpropGradient) / Math.abs(numericalGradient);
			
			
			if( relError > maxRelError ){
				if( print ) System.out.println("Param " + i + " FAILED: grad="+backpropGradient+", numericalGrad="+numericalGradient
						+ ", relError="+relError + ", scorePlus="+scorePlus+", scoreMinus="+scoreMinus);
				if( exitOnFirstError ) return false;
				if( relError > maxError ) maxError = relError;
				totalNFailures++;
			} else if( print ){
				System.out.println("Param " + i + " passed: grad="+backpropGradient+", numericalGrad="+numericalGradient
						+ ", relError="+relError );
			}
		}
		
		if( print ){
			int nPass = nParams - totalNFailures;
			System.out.println("GradientCheckUtil.checkGradients(): " + nParams + " params checked, " + nPass + " passed, " + totalNFailures + " failed. Largest relative error = " + maxError );
		}
		
		return totalNFailures == 0;
	}

}
