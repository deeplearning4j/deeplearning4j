package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.UpdaterCreator;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A utility for numerically checking gradients. <br>
 * Basic idea: compare calculated gradients with those calculated numerically,
 * to check implementation of backpropagation gradient calculation.<br>
 * See:<br>
 * - http://cs231n.github.io/neural-networks-3/#gradcheck<br>
 * - http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization<br>
 * - https://code.google.com/p/cuda-convnet/wiki/CheckingGradients<br>
 *
 *
 * Is C is cost function, then dC/dw ~= (C(w+epsilon)-C(w-epsilon)) / (2*epsilon).<br>
 * Method checks gradient calculation for every parameter separately by doing 2 forward pass
 * calculations for each parameter, so can be very time consuming for large networks.
 *
 * @author Alex Black
 */
public class GradientCheckUtil {

    private static Logger log = LoggerFactory.getLogger(GradientCheckUtil.class);

    /**
     * Check backprop gradients for a MultiLayerNetwork.
     * @param mln MultiLayerNetwork to test. This must be initialized.
     * @param epsilon Usually on the order/ of 1e-4 or so.
     * @param maxRelError Maximum relative error. Usually < 0.01, though maybe more for deep networks
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
                                          boolean print, boolean exitOnFirstError, INDArray input, INDArray labels, boolean useUpdater) {
        //Basic sanity checks on input:
        if(epsilon <= 0.0 || epsilon > 0.1)
            throw new IllegalArgumentException("Invalid epsilon: expect epsilon in range (0,0.1], usually 1e-4 or so");
        if(maxRelError <= 0.0 || maxRelError > 0.25)
            throw new IllegalArgumentException("Invalid maxRelativeError: " + maxRelError );
        if( !(mln.getOutputLayer() instanceof BaseOutputLayer))
            throw new IllegalArgumentException("Cannot check backprop gradients without OutputLayer");

        mln.setInput(input);
        mln.setLabels(labels);
        mln.computeGradientAndScore();
        Pair<Gradient,Double> gradAndScore = mln.gradientAndScore();

        if(useUpdater) {
            Updater updater = UpdaterCreator.getUpdater(mln);
            updater.update(mln, gradAndScore.getFirst(), 0, mln.batchSize());
        }

        INDArray gradientToCheck = gradAndScore.getFirst().gradient().dup();    //need dup: gradients are a *view* of the full gradient array (which will change every time backprop is done)
        INDArray originalParams = mln.params().dup();   //need dup: params are a *view* of full parameters

        int nParams = originalParams.length();

        int totalNFailures = 0;
        double maxError = 0.0;
        for(int i = 0; i < nParams; i++) {
            //(w+epsilon): Do forward pass and score
            INDArray params = originalParams.dup();
            params.putScalar(i, params.getDouble(i) + epsilon);
            mln.setParameters(params);
            mln.computeGradientAndScore();
            double scorePlus = mln.score();

            //(w-epsilon): Do forward pass and score
            params.putScalar(i, params.getDouble(i)  - 2*epsilon); // +eps - 2*eps = -eps
            mln.setParameters(params);
            mln.computeGradientAndScore();
            double scoreMinus = mln.score();

            //Calculate numerical parameter gradient:
            double scoreDelta = scorePlus - scoreMinus;

            double numericalGradient = scoreDelta / (2 * epsilon);
            if(Double.isNaN(numericalGradient))
                throw new IllegalStateException("Numerical gradient was NaN for parameter " + i + " of " + nParams);

            double backpropGradient = gradientToCheck.getDouble(i);
            //http://cs231n.github.io/neural-networks-3/#gradcheck
            //use mean centered
            double relError = Math.abs(backpropGradient - numericalGradient) / (Math.abs(numericalGradient) + Math.abs(backpropGradient));
            if( backpropGradient == 0.0 && numericalGradient == 0.0 ) relError = 0.0;	//Edge case: i.e., RNNs with time series length of 1.0

            if(relError > maxError) maxError = relError;
            if(relError > maxRelError || Double.isNaN(relError)) {
                if(print)
                    log.info("Param " + i + " FAILED: grad= " + backpropGradient + ", numericalGrad= "+numericalGradient
                            + ", relError= " + relError + ", scorePlus="+scorePlus+", scoreMinus= " + scoreMinus);
                if(exitOnFirstError)
                    return false;
                totalNFailures++;
            }
            else if(print) {
                log.info("Param " + i + " passed: grad= " + backpropGradient + ", numericalGrad= " + numericalGradient
                        + ", relError= " + relError );
            }
        }

        if(print) {
            int nPass = nParams - totalNFailures;
            log.info("GradientCheckUtil.checkGradients(): " + nParams + " params checked, " + nPass + " passed, " + totalNFailures + " failed. Largest relative error = " + maxError );
        }

        return totalNFailures == 0;
    }





    /**Check backprop gradients for a ComputationGraph
     * @param graph ComputationGraph to test. This must be initialized.
     * @param epsilon Usually on the order of 1e-4 or so.
     * @param maxRelError Maximum relative error. Usually < 0.01, though maybe more for deep networks
     * @param print Whether to print full pass/failure details for each parameter gradient
     * @param exitOnFirstError If true: return upon first failure. If false: continue checking even if
     *  one parameter gradient has failed. Typically use false for debugging, true for unit tests.
     * @param inputs Input arrays to use for forward pass. May be mini-batch data.
     * @param labels Labels/targets (output) arrays to use to calculate backprop gradient. May be mini-batch data.
     * @return true if gradients are passed, false otherwise.
     */
    public static boolean checkGradients( ComputationGraph graph, double epsilon, double maxRelError,
                                          boolean print, boolean exitOnFirstError, INDArray[] inputs, INDArray[] labels) {
        //Basic sanity checks on input:
        if(epsilon <= 0.0 || epsilon > 0.1)
            throw new IllegalArgumentException("Invalid epsilon: expect epsilon in range (0,0.1], usually 1e-4 or so");
        if(maxRelError <= 0.0 || maxRelError > 0.25)
            throw new IllegalArgumentException("Invalid maxRelativeError: " + maxRelError );

        if(graph.getNumInputArrays() != inputs.length) throw new IllegalArgumentException("Invalid input arrays: expect " + graph.getNumInputArrays() + " inputs");
        if(graph.getNumOutputArrays() != labels.length) throw new IllegalArgumentException("Invalid labels arrays: expect " + graph.getNumOutputArrays() + " outputs");

        for( int i=0; i<inputs.length; i++ ) graph.setInput(i,inputs[i]);
        for( int i=0; i<labels.length; i++ ) graph.setLabel(i,labels[i]);

        graph.computeGradientAndScore();
        Pair<Gradient,Double> gradAndScore = graph.gradientAndScore();

        ComputationGraphUpdater updater = new ComputationGraphUpdater(graph);
        updater.update(graph, gradAndScore.getFirst(), 0, graph.batchSize());

        INDArray gradientToCheck = gradAndScore.getFirst().gradient().dup();    //need dup: gradients are a *view* of the full gradient array (which will change every time backprop is done)
        INDArray originalParams = graph.params().dup();   //need dup: params are a *view* of full parameters

        int nParams = originalParams.length();

        int totalNFailures = 0;
        double maxError = 0.0;
        for(int i = 0; i < nParams; i++) {
            //(w+epsilon): Do forward pass and score
            INDArray params = originalParams.dup();
            params.putScalar(i, params.getDouble(i) + epsilon);
            graph.setParams(params);
            graph.computeGradientAndScore();
            double scorePlus = graph.score();

            //(w-epsilon): Do forward pass and score
            params.putScalar(i, params.getDouble(i)  - 2*epsilon); // +eps - 2*eps = -eps
            graph.setParams(params);
            graph.computeGradientAndScore();
            double scoreMinus = graph.score();

            //Calculate numerical parameter gradient:
            double scoreDelta = scorePlus - scoreMinus;

            double numericalGradient = scoreDelta / (2 * epsilon);
            if(Double.isNaN(numericalGradient))
                throw new IllegalStateException("Numerical gradient was NaN for parameter " + i + " of " + nParams);

            double backpropGradient = gradientToCheck.getDouble(i);
            //http://cs231n.github.io/neural-networks-3/#gradcheck
            //use mean centered
            double relError = Math.abs(backpropGradient - numericalGradient) / (Math.abs(numericalGradient) + Math.abs(backpropGradient));
            if( backpropGradient == 0.0 && numericalGradient == 0.0 ) relError = 0.0;	//Edge case: i.e., RNNs with time series length of 1.0

            if(relError > maxError) maxError = relError;
            if(relError > maxRelError || Double.isNaN(relError)) {
                if(print)
                    log.info("Param " + i + " FAILED: grad= " + backpropGradient + ", numericalGrad= "+numericalGradient
                            + ", relError= " + relError + ", scorePlus="+scorePlus+", scoreMinus= " + scoreMinus);
                if(exitOnFirstError)
                    return false;
                totalNFailures++;
            }
            else if(print) {
                log.info("Param " + i + " passed: grad= " + backpropGradient + ", numericalGrad= " + numericalGradient
                        + ", relError= " + relError );
            }
        }

        if(print) {
            int nPass = nParams - totalNFailures;
            log.info("GradientCheckUtil.checkGradients(): " + nParams + " params checked, " + nPass + " passed, " + totalNFailures + " failed. Largest relative error = " + maxError );
        }

        return totalNFailures == 0;
    }

}
