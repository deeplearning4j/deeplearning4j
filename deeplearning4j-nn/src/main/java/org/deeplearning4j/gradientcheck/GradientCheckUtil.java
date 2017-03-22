package org.deeplearning4j.gradientcheck;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.UpdaterCreator;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

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

    private static final List<Class<? extends IActivation>> VALID_ACTIVATION_FUNCTIONS =
                    Arrays.asList(Activation.CUBE.getActivationFunction().getClass(),
                                    Activation.ELU.getActivationFunction().getClass(),
                                    Activation.IDENTITY.getActivationFunction().getClass(),
                                    Activation.RATIONALTANH.getActivationFunction().getClass(),
                                    Activation.SIGMOID.getActivationFunction().getClass(),
                                    Activation.SOFTMAX.getActivationFunction().getClass(),
                                    Activation.SOFTPLUS.getActivationFunction().getClass(),
                                    Activation.SOFTSIGN.getActivationFunction().getClass(),
                                    Activation.TANH.getActivationFunction().getClass());

    private GradientCheckUtil() {}

    /**
     * Check backprop gradients for a MultiLayerNetwork.
     * @param mln MultiLayerNetwork to test. This must be initialized.
     * @param epsilon Usually on the order/ of 1e-4 or so.
     * @param maxRelError Maximum relative error. Usually < 1e-5 or so, though maybe more for deep networks or those with nonlinear activation
     * @param minAbsoluteError Minimum absolute error to cause a failure. Numerical gradients can be non-zero due to precision issues.
     *                         For example, 0.0 vs. 1e-18: relative error is 1.0, but not really a failure
     * @param print Whether to print full pass/failure details for each parameter gradient
     * @param exitOnFirstError If true: return upon first failure. If false: continue checking even if
     *  one parameter gradient has failed. Typically use false for debugging, true for unit tests.
     * @param input Input array to use for forward pass. May be mini-batch data.
     * @param labels Labels/targets to use to calculate backprop gradient. May be mini-batch data.
     * @return true if gradients are passed, false otherwise.
     */
    public static boolean checkGradients(MultiLayerNetwork mln, double epsilon, double maxRelError,
                    double minAbsoluteError, boolean print, boolean exitOnFirstError, INDArray input, INDArray labels) {
        //Basic sanity checks on input:
        if (epsilon <= 0.0 || epsilon > 0.1)
            throw new IllegalArgumentException("Invalid epsilon: expect epsilon in range (0,0.1], usually 1e-4 or so");
        if (maxRelError <= 0.0 || maxRelError > 0.25)
            throw new IllegalArgumentException("Invalid maxRelativeError: " + maxRelError);
        if (!(mln.getOutputLayer() instanceof IOutputLayer))
            throw new IllegalArgumentException("Cannot check backprop gradients without OutputLayer");

        //Check network configuration:

        int layerCount = 0;
        for (NeuralNetConfiguration n : mln.getLayerWiseConfigurations().getConfs()) {
            org.deeplearning4j.nn.conf.Updater u = n.getLayer().getUpdater();
            if (u == org.deeplearning4j.nn.conf.Updater.SGD) {
                //Must have LR of 1.0
                double lr = n.getLayer().getLearningRate();
                if (lr != 1.0) {
                    throw new IllegalStateException("When using SGD updater, must also use lr=1.0 for layer "
                                    + layerCount + "; got " + u + " with lr=" + lr + " for layer \""
                                    + n.getLayer().getLayerName() + "\"");
                }
            } else if (u != org.deeplearning4j.nn.conf.Updater.NONE) {
                throw new IllegalStateException(
                                "Must have Updater.NONE (or SGD + lr=1.0) for layer " + layerCount + "; got " + u);
            }

            double dropout = n.getLayer().getDropOut();
            if (n.isUseRegularization() && dropout != 0.0) {
                throw new IllegalStateException("Must have dropout == 0.0 for gradient checks - got dropout = "
                                + dropout + " for layer " + layerCount);
            }

            IActivation activation = n.getLayer().getActivationFn();
            if (activation != null) {
                if (!VALID_ACTIVATION_FUNCTIONS.contains(activation.getClass())) {
                    log.warn("Layer " + layerCount + " is possibly using an unsuitable activation function: "
                                    + activation.getClass()
                                    + ". Activation functions for gradient checks must be smooth (like sigmoid, tanh, softmax) and not "
                                    + "contain discontinuities like ReLU or LeakyReLU (these may cause spurious failures)");
                }
            }
        }

        mln.setInput(input);
        mln.setLabels(labels);
        mln.computeGradientAndScore();
        Pair<Gradient, Double> gradAndScore = mln.gradientAndScore();

        Updater updater = UpdaterCreator.getUpdater(mln);
        updater.update(mln, gradAndScore.getFirst(), 0, mln.batchSize());

        INDArray gradientToCheck = gradAndScore.getFirst().gradient().dup(); //need dup: gradients are a *view* of the full gradient array (which will change every time backprop is done)
        INDArray originalParams = mln.params().dup(); //need dup: params are a *view* of full parameters

        int nParams = originalParams.length();

        Map<String, INDArray> paramTable = mln.paramTable();
        List<String> paramNames = new ArrayList<>(paramTable.keySet());
        int[] paramEnds = new int[paramNames.size()];
        paramEnds[0] = paramTable.get(paramNames.get(0)).length();
        for (int i = 1; i < paramEnds.length; i++) {
            paramEnds[i] = paramEnds[i - 1] + paramTable.get(paramNames.get(i)).length();
        }


        int totalNFailures = 0;
        double maxError = 0.0;
        DataSet ds = new DataSet(input, labels);
        int currParamNameIdx = 0;

        INDArray params = mln.params(); //Assumption here: params is a view that we can modify in-place
        for (int i = 0; i < nParams; i++) {
            //Get param name
            if (i >= paramEnds[currParamNameIdx]) {
                currParamNameIdx++;
            }
            String paramName = paramNames.get(currParamNameIdx);

            //(w+epsilon): Do forward pass and score
            double origValue = params.getDouble(i);
            params.putScalar(i, origValue + epsilon);
            double scorePlus = mln.score(ds, true);

            //(w-epsilon): Do forward pass and score
            params.putScalar(i, origValue - epsilon);
            double scoreMinus = mln.score(ds, true);

            //Reset original param value
            params.putScalar(i, origValue);

            //Calculate numerical parameter gradient:
            double scoreDelta = scorePlus - scoreMinus;

            double numericalGradient = scoreDelta / (2 * epsilon);
            if (Double.isNaN(numericalGradient))
                throw new IllegalStateException("Numerical gradient was NaN for parameter " + i + " of " + nParams);

            double backpropGradient = gradientToCheck.getDouble(i);
            //http://cs231n.github.io/neural-networks-3/#gradcheck
            //use mean centered
            double relError = Math.abs(backpropGradient - numericalGradient)
                            / (Math.abs(numericalGradient) + Math.abs(backpropGradient));
            if (backpropGradient == 0.0 && numericalGradient == 0.0)
                relError = 0.0; //Edge case: i.e., RNNs with time series length of 1.0

            if (relError > maxError)
                maxError = relError;
            if (relError > maxRelError || Double.isNaN(relError)) {
                double absError = Math.abs(backpropGradient - numericalGradient);
                if (absError < minAbsoluteError) {
                    log.info("Param " + i + " (" + paramName + ") passed: grad= " + backpropGradient
                                    + ", numericalGrad= " + numericalGradient + ", relError= " + relError
                                    + "; absolute error = " + absError + " < minAbsoluteError = " + minAbsoluteError);
                } else {
                    if (print)
                        log.info("Param " + i + " (" + paramName + ") FAILED: grad= " + backpropGradient
                                        + ", numericalGrad= " + numericalGradient + ", relError= " + relError
                                        + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus);
                    if (exitOnFirstError)
                        return false;
                    totalNFailures++;
                }
            } else if (print) {
                log.info("Param " + i + " (" + paramName + ") passed: grad= " + backpropGradient + ", numericalGrad= "
                                + numericalGradient + ", relError= " + relError);
            }
        }

        if (print) {
            int nPass = nParams - totalNFailures;
            log.info("GradientCheckUtil.checkGradients(): " + nParams + " params checked, " + nPass + " passed, "
                            + totalNFailures + " failed. Largest relative error = " + maxError);
        }

        return totalNFailures == 0;
    }



    /**Check backprop gradients for a ComputationGraph
     * @param graph ComputationGraph to test. This must be initialized.
     * @param epsilon Usually on the order of 1e-4 or so.
     * @param maxRelError Maximum relative error. Usually < 0.01, though maybe more for deep networks
     * @param minAbsoluteError Minimum absolute error to cause a failure. Numerical gradients can be non-zero due to precision issues.
     *                         For example, 0.0 vs. 1e-18: relative error is 1.0, but not really a failure
     * @param print Whether to print full pass/failure details for each parameter gradient
     * @param exitOnFirstError If true: return upon first failure. If false: continue checking even if
     *  one parameter gradient has failed. Typically use false for debugging, true for unit tests.
     * @param inputs Input arrays to use for forward pass. May be mini-batch data.
     * @param labels Labels/targets (output) arrays to use to calculate backprop gradient. May be mini-batch data.
     * @return true if gradients are passed, false otherwise.
     */
    public static boolean checkGradients(ComputationGraph graph, double epsilon, double maxRelError,
                    double minAbsoluteError, boolean print, boolean exitOnFirstError, INDArray[] inputs,
                    INDArray[] labels) {
        //Basic sanity checks on input:
        if (epsilon <= 0.0 || epsilon > 0.1)
            throw new IllegalArgumentException("Invalid epsilon: expect epsilon in range (0,0.1], usually 1e-4 or so");
        if (maxRelError <= 0.0 || maxRelError > 0.25)
            throw new IllegalArgumentException("Invalid maxRelativeError: " + maxRelError);

        if (graph.getNumInputArrays() != inputs.length)
            throw new IllegalArgumentException("Invalid input arrays: expect " + graph.getNumInputArrays() + " inputs");
        if (graph.getNumOutputArrays() != labels.length)
            throw new IllegalArgumentException(
                            "Invalid labels arrays: expect " + graph.getNumOutputArrays() + " outputs");

        //Check configuration
        int layerCount = 0;
        for (String vertexName : graph.getConfiguration().getVertices().keySet()) {
            GraphVertex gv = graph.getConfiguration().getVertices().get(vertexName);
            if (!(gv instanceof LayerVertex))
                continue;
            LayerVertex lv = (LayerVertex) gv;

            org.deeplearning4j.nn.conf.Updater u = lv.getLayerConf().getLayer().getUpdater();
            if (u == org.deeplearning4j.nn.conf.Updater.SGD) {
                //Must have LR of 1.0
                double lr = lv.getLayerConf().getLayer().getLearningRate();
                if (lr != 1.0) {
                    throw new IllegalStateException("When using SGD updater, must also use lr=1.0 for layer \""
                                    + vertexName + "\"; got " + u);
                }
            } else if (u != org.deeplearning4j.nn.conf.Updater.NONE) {
                throw new IllegalStateException(
                                "Must have Updater.NONE (or SGD + lr=1.0) for layer \"" + vertexName + "\"; got " + u);
            }

            double dropout = lv.getLayerConf().getLayer().getDropOut();
            if (lv.getLayerConf().isUseRegularization() && dropout != 0.0) {
                throw new IllegalStateException("Must have dropout == 0.0 for gradient checks - got dropout = "
                                + dropout + " for layer " + layerCount);
            }

            IActivation activation = lv.getLayerConf().getLayer().getActivationFn();
            if (activation != null) {
                if (!VALID_ACTIVATION_FUNCTIONS.contains(activation.getClass())) {
                    log.warn("Layer \"" + vertexName + "\" is possibly using an unsuitable activation function: "
                                    + activation.getClass()
                                    + ". Activation functions for gradient checks must be smooth (like sigmoid, tanh, softmax) and not "
                                    + "contain discontinuities like ReLU or LeakyReLU (these may cause spurious failures)");
                }
            }
        }

        for (int i = 0; i < inputs.length; i++)
            graph.setInput(i, inputs[i]);
        for (int i = 0; i < labels.length; i++)
            graph.setLabel(i, labels[i]);

        graph.computeGradientAndScore();
        Pair<Gradient, Double> gradAndScore = graph.gradientAndScore();

        ComputationGraphUpdater updater = new ComputationGraphUpdater(graph);
        updater.update(graph, gradAndScore.getFirst(), 0, graph.batchSize());

        INDArray gradientToCheck = gradAndScore.getFirst().gradient().dup(); //need dup: gradients are a *view* of the full gradient array (which will change every time backprop is done)
        INDArray originalParams = graph.params().dup(); //need dup: params are a *view* of full parameters

        int nParams = originalParams.length();

        Map<String, INDArray> paramTable = graph.paramTable();
        List<String> paramNames = new ArrayList<>(paramTable.keySet());
        int[] paramEnds = new int[paramNames.size()];
        paramEnds[0] = paramTable.get(paramNames.get(0)).length();
        for (int i = 1; i < paramEnds.length; i++) {
            paramEnds[i] = paramEnds[i - 1] + paramTable.get(paramNames.get(i)).length();
        }

        int currParamNameIdx = 0;
        int totalNFailures = 0;
        double maxError = 0.0;
        MultiDataSet mds = new MultiDataSet(inputs, labels);
        INDArray params = graph.params(); //Assumption here: params is a view that we can modify in-place
        for (int i = 0; i < nParams; i++) {
            //Get param name
            if (i >= paramEnds[currParamNameIdx]) {
                currParamNameIdx++;
            }
            String paramName = paramNames.get(currParamNameIdx);

            //(w+epsilon): Do forward pass and score
            double origValue = params.getDouble(i);

            params.putScalar(i, origValue + epsilon);
            double scorePlus = graph.score(mds, true); //training == true for batch norm, etc (scores and gradients need to be calculated on same thing)

            //(w-epsilon): Do forward pass and score
            params.putScalar(i, origValue - epsilon);
            double scoreMinus = graph.score(mds, true);

            //Reset original param value
            params.putScalar(i, origValue);

            //Calculate numerical parameter gradient:
            double scoreDelta = scorePlus - scoreMinus;

            double numericalGradient = scoreDelta / (2 * epsilon);
            if (Double.isNaN(numericalGradient))
                throw new IllegalStateException("Numerical gradient was NaN for parameter " + i + " of " + nParams);

            double backpropGradient = gradientToCheck.getDouble(i);
            //http://cs231n.github.io/neural-networks-3/#gradcheck
            //use mean centered
            double relError = Math.abs(backpropGradient - numericalGradient)
                            / (Math.abs(numericalGradient) + Math.abs(backpropGradient));
            if (backpropGradient == 0.0 && numericalGradient == 0.0)
                relError = 0.0; //Edge case: i.e., RNNs with time series length of 1.0

            if (relError > maxError)
                maxError = relError;
            if (relError > maxRelError || Double.isNaN(relError)) {
                double absError = Math.abs(backpropGradient - numericalGradient);
                if (absError < minAbsoluteError) {
                    log.info("Param " + i + " (" + paramName + ") passed: grad= " + backpropGradient
                                    + ", numericalGrad= " + numericalGradient + ", relError= " + relError
                                    + "; absolute error = " + absError + " < minAbsoluteError = " + minAbsoluteError);
                } else {
                    if (print)
                        log.info("Param " + i + " (" + paramName + ") FAILED: grad= " + backpropGradient
                                        + ", numericalGrad= " + numericalGradient + ", relError= " + relError
                                        + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus);
                    if (exitOnFirstError)
                        return false;
                    totalNFailures++;
                }
            } else if (print) {
                log.info("Param " + i + " (" + paramName + ") passed: grad= " + backpropGradient + ", numericalGrad= "
                                + numericalGradient + ", relError= " + relError);
            }
        }

        if (print) {
            int nPass = nParams - totalNFailures;
            log.info("GradientCheckUtil.checkGradients(): " + nParams + " params checked, " + nPass + " passed, "
                            + totalNFailures + " failed. Largest relative error = " + maxError);
        }

        return totalNFailures == 0;
    }



    /**
     * Check backprop gradients for a pretrain layer
     *
     * NOTE: gradient checking pretrain layers can be difficult...
     */
    public static boolean checkGradientsPretrainLayer(Layer layer, double epsilon, double maxRelError,
                    double minAbsoluteError, boolean print, boolean exitOnFirstError, INDArray input, int rngSeed) {
        //Basic sanity checks on input:
        if (epsilon <= 0.0 || epsilon > 0.1)
            throw new IllegalArgumentException("Invalid epsilon: expect epsilon in range (0,0.1], usually 1e-4 or so");
        if (maxRelError <= 0.0 || maxRelError > 0.25)
            throw new IllegalArgumentException("Invalid maxRelativeError: " + maxRelError);

        //Check network configuration:

        int layerCount = 0;

        layer.setInput(input);
        Nd4j.getRandom().setSeed(rngSeed);
        layer.computeGradientAndScore();
        Pair<Gradient, Double> gradAndScore = layer.gradientAndScore();

        Updater updater = UpdaterCreator.getUpdater(layer);
        updater.update(layer, gradAndScore.getFirst(), 0, layer.batchSize());

        INDArray gradientToCheck = gradAndScore.getFirst().gradient().dup(); //need dup: gradients are a *view* of the full gradient array (which will change every time backprop is done)
        INDArray originalParams = layer.params().dup(); //need dup: params are a *view* of full parameters

        int nParams = originalParams.length();

        Map<String, INDArray> paramTable = layer.paramTable();
        List<String> paramNames = new ArrayList<>(paramTable.keySet());
        int[] paramEnds = new int[paramNames.size()];
        paramEnds[0] = paramTable.get(paramNames.get(0)).length();
        for (int i = 1; i < paramEnds.length; i++) {
            paramEnds[i] = paramEnds[i - 1] + paramTable.get(paramNames.get(i)).length();
        }


        int totalNFailures = 0;
        double maxError = 0.0;
        int currParamNameIdx = 0;

        INDArray params = layer.params(); //Assumption here: params is a view that we can modify in-place
        for (int i = 0; i < nParams; i++) {
            //Get param name
            if (i >= paramEnds[currParamNameIdx]) {
                currParamNameIdx++;
            }
            String paramName = paramNames.get(currParamNameIdx);

            //(w+epsilon): Do forward pass and score
            double origValue = params.getDouble(i);
            params.putScalar(i, origValue + epsilon);

            //TODO add a 'score' method that doesn't calculate gradients...
            Nd4j.getRandom().setSeed(rngSeed);
            layer.computeGradientAndScore();
            double scorePlus = layer.score();

            //(w-epsilon): Do forward pass and score
            params.putScalar(i, origValue - epsilon);
            Nd4j.getRandom().setSeed(rngSeed);
            layer.computeGradientAndScore();
            double scoreMinus = layer.score();

            //Reset original param value
            params.putScalar(i, origValue);

            //Calculate numerical parameter gradient:
            double scoreDelta = scorePlus - scoreMinus;

            double numericalGradient = scoreDelta / (2 * epsilon);
            if (Double.isNaN(numericalGradient))
                throw new IllegalStateException("Numerical gradient was NaN for parameter " + i + " of " + nParams);

            double backpropGradient = gradientToCheck.getDouble(i);
            //http://cs231n.github.io/neural-networks-3/#gradcheck
            //use mean centered
            double relError = Math.abs(backpropGradient - numericalGradient)
                            / (Math.abs(numericalGradient) + Math.abs(backpropGradient));
            if (backpropGradient == 0.0 && numericalGradient == 0.0)
                relError = 0.0; //Edge case: i.e., RNNs with time series length of 1.0

            if (relError > maxError)
                maxError = relError;
            if (relError > maxRelError || Double.isNaN(relError)) {
                double absError = Math.abs(backpropGradient - numericalGradient);
                if (absError < minAbsoluteError) {
                    log.info("Param " + i + " (" + paramName + ") passed: grad= " + backpropGradient
                                    + ", numericalGrad= " + numericalGradient + ", relError= " + relError
                                    + "; absolute error = " + absError + " < minAbsoluteError = " + minAbsoluteError);
                } else {
                    if (print)
                        log.info("Param " + i + " (" + paramName + ") FAILED: grad= " + backpropGradient
                                        + ", numericalGrad= " + numericalGradient + ", relError= " + relError
                                        + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus);
                    if (exitOnFirstError)
                        return false;
                    totalNFailures++;
                }
            } else if (print) {
                log.info("Param " + i + " (" + paramName + ") passed: grad= " + backpropGradient + ", numericalGrad= "
                                + numericalGradient + ", relError= " + relError);
            }
        }

        if (print) {
            int nPass = nParams - totalNFailures;
            log.info("GradientCheckUtil.checkGradients(): " + nParams + " params checked, " + nPass + " passed, "
                            + totalNFailures + " failed. Largest relative error = " + maxError);
        }

        return totalNFailures == 0;
    }

}
