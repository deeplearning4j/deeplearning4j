package org.deeplearning4j.gradientcheck;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.layers.LossLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.UpdaterCreator;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.*;

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
@Slf4j
public class GradientCheckUtil {

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


    private static void configureLossFnClippingIfPresent(IOutputLayer outputLayer){

        ILossFunction lfn = null;
        IActivation afn = null;
        if(outputLayer instanceof BaseOutputLayer){
            BaseOutputLayer o = (BaseOutputLayer)outputLayer;
            lfn = ((org.deeplearning4j.nn.conf.layers.BaseOutputLayer)o.layerConf()).getLossFn();
            afn = o.layerConf().getActivationFn();
        } else if(outputLayer instanceof LossLayer){
            LossLayer o = (LossLayer) outputLayer;
            lfn = o.layerConf().getLossFn();
            afn = o.layerConf().getActivationFn();
        }

        if (lfn instanceof LossMCXENT && afn instanceof ActivationSoftmax && ((LossMCXENT) lfn).getSoftmaxClipEps() != 0) {
            log.info("Setting softmax clipping epsilon to 0.0 for " + lfn.getClass()
                    + " loss function to avoid spurious gradient check failures");
            ((LossMCXENT) lfn).setSoftmaxClipEps(0.0);
        } else if(lfn instanceof LossBinaryXENT && ((LossBinaryXENT) lfn).getClipEps() != 0) {
            log.info("Setting clipping epsilon to 0.0 for " + lfn.getClass()
                    + " loss function to avoid spurious gradient check failures");
            ((LossBinaryXENT) lfn).setClipEps(0.0);
        }
    }

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
        return checkGradients(mln, epsilon, maxRelError, minAbsoluteError, print, exitOnFirstError, input, labels, null, null);
    }

    public static boolean checkGradients(MultiLayerNetwork mln, double epsilon, double maxRelError,
                                         double minAbsoluteError, boolean print, boolean exitOnFirstError,
                                         INDArray input, INDArray labels, INDArray inputMask, INDArray labelMask) {
        return checkGradients(mln, epsilon, maxRelError, minAbsoluteError, print, exitOnFirstError,
                input, labels, inputMask, labelMask, false, -1);
    }

    public static boolean checkGradients(MultiLayerNetwork mln, double epsilon, double maxRelError,
                                         double minAbsoluteError, boolean print, boolean exitOnFirstError,
                                         INDArray input, INDArray labels, INDArray inputMask, INDArray labelMask,
                                         boolean subset, int maxPerParam) {
        return checkGradients(mln, epsilon, maxRelError, minAbsoluteError, print, exitOnFirstError, input,
                labels, inputMask, labelMask, subset, maxPerParam, null);
    }

    public static boolean checkGradients(MultiLayerNetwork mln, double epsilon, double maxRelError,
                                         double minAbsoluteError, boolean print, boolean exitOnFirstError,
                                         INDArray input, INDArray labels, INDArray inputMask, INDArray labelMask,
                                         boolean subset, int maxPerParam, Set<String> excludeParams) {
        //Basic sanity checks on input:
        if (epsilon <= 0.0 || epsilon > 0.1)
            throw new IllegalArgumentException("Invalid epsilon: expect epsilon in range (0,0.1], usually 1e-4 or so");
        if (maxRelError <= 0.0 || maxRelError > 0.25)
            throw new IllegalArgumentException("Invalid maxRelativeError: " + maxRelError);
        if (!(mln.getOutputLayer() instanceof IOutputLayer))
            throw new IllegalArgumentException("Cannot check backprop gradients without OutputLayer");

        DataBuffer.Type dataType = DataTypeUtil.getDtypeFromContext();
        if (dataType != DataBuffer.Type.DOUBLE) {
            throw new IllegalStateException("Cannot perform gradient check: Datatype is not set to double precision ("
                            + "is: " + dataType + "). Double precision must be used for gradient checks. Set "
                            + "DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE); before using GradientCheckUtil");
        }

        //Check network configuration:
        int layerCount = 0;
        for (NeuralNetConfiguration n : mln.getLayerWiseConfigurations().getConfs()) {
            if (n.getLayer() instanceof BaseLayer) {
                BaseLayer bl = (BaseLayer) n.getLayer();
                IUpdater u = bl.getIUpdater();
                if (u instanceof Sgd) {
                    //Must have LR of 1.0
                    double lr = ((Sgd) u).getLearningRate();
                    if (lr != 1.0) {
                        throw new IllegalStateException("When using SGD updater, must also use lr=1.0 for layer "
                                        + layerCount + "; got " + u + " with lr=" + lr + " for layer \""
                                        + n.getLayer().getLayerName() + "\"");
                    }
                } else if (!(u instanceof NoOp)) {
                    throw new IllegalStateException(
                                    "Must have Updater.NONE (or SGD + lr=1.0) for layer " + layerCount + "; got " + u);
                }

                IActivation activation = bl.getActivationFn();
                if (activation != null) {
                    if (!VALID_ACTIVATION_FUNCTIONS.contains(activation.getClass())) {
                        log.warn("Layer " + layerCount + " is possibly using an unsuitable activation function: "
                                        + activation.getClass()
                                        + ". Activation functions for gradient checks must be smooth (like sigmoid, tanh, softmax) and not "
                                        + "contain discontinuities like ReLU or LeakyReLU (these may cause spurious failures)");
                    }
                }
            }

            if (n.getLayer().getIDropout() != null) {
                throw new IllegalStateException("Must have no dropout for gradient checks - got dropout = "
                                + n.getLayer().getIDropout() + " for layer " + layerCount);
            }
        }

        //Set softmax clipping to 0 if necessary, to avoid spurious failures due to clipping
        for(Layer l : mln.getLayers()){
            if(l instanceof IOutputLayer){
                configureLossFnClippingIfPresent((IOutputLayer) l);
            }
        }

        mln.setInput(input);
        mln.setLabels(labels);
        mln.setLayerMaskArrays(inputMask, labelMask);
        mln.computeGradientAndScore();
        Pair<Gradient, Double> gradAndScore = mln.gradientAndScore();

        Updater updater = UpdaterCreator.getUpdater(mln);
        updater.update(mln, gradAndScore.getFirst(), 0, 0, mln.batchSize(), LayerWorkspaceMgr.noWorkspaces());

        INDArray gradientToCheck = gradAndScore.getFirst().gradient().dup(); //need dup: gradients are a *view* of the full gradient array (which will change every time backprop is done)
        INDArray originalParams = mln.params().dup(); //need dup: params are a *view* of full parameters

        int nParams = originalParams.length();

        Map<String, INDArray> paramTable = mln.paramTable();
        List<String> paramNames = new ArrayList<>(paramTable.keySet());
        int[] paramEnds = new int[paramNames.size()];
        paramEnds[0] = paramTable.get(paramNames.get(0)).length();
        Map<String,Integer> stepSizeForParam;
        if(subset){
            stepSizeForParam = new HashMap<>();
            stepSizeForParam.put(paramNames.get(0), Math.max(1, paramTable.get(paramNames.get(0)).length() / maxPerParam));
        } else {
            stepSizeForParam = null;
        }
        for (int i = 1; i < paramEnds.length; i++) {
            int n = paramTable.get(paramNames.get(i)).length();
            paramEnds[i] = paramEnds[i - 1] + n;
            if(subset){
                int ss = n / maxPerParam;
                if(ss == 0){
                    ss = n;
                }
                stepSizeForParam.put(paramNames.get(i), ss);
            }
        }

        if(print) {
            int i=0;
            for (Layer l : mln.getLayers()) {
                Set<String> s = l.paramTable().keySet();
                log.info("Layer " + i + ": " + l.getClass().getSimpleName() + " - params " + s);
                i++;
            }
        }


        int totalNFailures = 0;
        double maxError = 0.0;
        DataSet ds = new DataSet(input, labels, inputMask, labelMask);
        int currParamNameIdx = 0;

        INDArray params = mln.params(); //Assumption here: params is a view that we can modify in-place
        for (int i = 0; i < nParams; ) {
            //Get param name
            if (i >= paramEnds[currParamNameIdx]) {
                currParamNameIdx++;
            }
            String paramName = paramNames.get(currParamNameIdx);
            if(excludeParams != null && excludeParams.contains(paramName)){
                log.info("Skipping parameters for parameter name: {}", paramName);
                i = paramEnds[currParamNameIdx++];
                continue;
            }

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
                    if(print) {
                        log.info("Param " + i + " (" + paramName + ") passed: grad= " + backpropGradient
                                + ", numericalGrad= " + numericalGradient + ", relError= " + relError
                                + "; absolute error = " + absError + " < minAbsoluteError = " + minAbsoluteError);
                    }
                } else {
                    if (print)
                        log.info("Param " + i + " (" + paramName + ") FAILED: grad= " + backpropGradient
                                        + ", numericalGrad= " + numericalGradient + ", relError= " + relError
                                        + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus + ", paramValue = " + origValue);
                    if (exitOnFirstError)
                        return false;
                    totalNFailures++;
                }
            } else if (print) {
                log.info("Param " + i + " (" + paramName + ") passed: grad= " + backpropGradient + ", numericalGrad= "
                                + numericalGradient + ", relError= " + relError);
            }

            int step;
            if(subset){
                step = stepSizeForParam.get(paramName);
                if(i + step > paramEnds[currParamNameIdx]+1){
                    step = paramEnds[currParamNameIdx]+1 - i;
                }
            } else {
                step = 1;
            }

            i += step;
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
        return checkGradients(graph, epsilon, maxRelError, minAbsoluteError, print, exitOnFirstError, inputs, labels, null, null, null);
    }

    public static boolean checkGradients(ComputationGraph graph, double epsilon, double maxRelError,
                                         double minAbsoluteError, boolean print, boolean exitOnFirstError, INDArray[] inputs,
                                         INDArray[] labels, INDArray[] fMask, INDArray[] lMask) {
        return checkGradients(graph, epsilon, maxRelError, minAbsoluteError, print, exitOnFirstError, inputs,
                labels, fMask, lMask, null);
    }

    public static boolean checkGradients(ComputationGraph graph, double epsilon, double maxRelError,
                                         double minAbsoluteError, boolean print, boolean exitOnFirstError, INDArray[] inputs,
                                         INDArray[] labels, INDArray[] fMask, INDArray[] lMask, Set<String> excludeParams) {
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

        DataBuffer.Type dataType = DataTypeUtil.getDtypeFromContext();
        if (dataType != DataBuffer.Type.DOUBLE) {
            throw new IllegalStateException("Cannot perform gradient check: Datatype is not set to double precision ("
                            + "is: " + dataType + "). Double precision must be used for gradient checks. Set "
                            + "DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE); before using GradientCheckUtil");
        }

        //Check configuration
        int layerCount = 0;
        for (String vertexName : graph.getConfiguration().getVertices().keySet()) {
            GraphVertex gv = graph.getConfiguration().getVertices().get(vertexName);
            if (!(gv instanceof LayerVertex))
                continue;
            LayerVertex lv = (LayerVertex) gv;

            if (lv.getLayerConf().getLayer() instanceof BaseLayer) {
                BaseLayer bl = (BaseLayer) lv.getLayerConf().getLayer();
                IUpdater u = bl.getIUpdater();
                if (u instanceof Sgd) {
                    //Must have LR of 1.0
                    double lr = ((Sgd) u).getLearningRate();
                    if (lr != 1.0) {
                        throw new IllegalStateException("When using SGD updater, must also use lr=1.0 for layer "
                                        + layerCount + "; got " + u + " with lr=" + lr + " for layer \""
                                        + lv.getLayerConf().getLayer().getLayerName() + "\"");
                    }
                } else if (!(u instanceof NoOp)) {
                    throw new IllegalStateException(
                                    "Must have Updater.NONE (or SGD + lr=1.0) for layer " + layerCount + "; got " + u);
                }

                IActivation activation = bl.getActivationFn();
                if (activation != null) {
                    if (!VALID_ACTIVATION_FUNCTIONS.contains(activation.getClass())) {
                        log.warn("Layer \"" + vertexName + "\" is possibly using an unsuitable activation function: "
                                        + activation.getClass()
                                        + ". Activation functions for gradient checks must be smooth (like sigmoid, tanh, softmax) and not "
                                        + "contain discontinuities like ReLU or LeakyReLU (these may cause spurious failures)");
                    }
                }
            }

            if (lv.getLayerConf().getLayer().getIDropout() != null) {
                throw new IllegalStateException("Must have no dropout for gradient checks - got dropout = "
                        + lv.getLayerConf().getLayer().getIDropout() + " for layer " + layerCount);
            }
        }

        //Set softmax clipping to 0 if necessary, to avoid spurious failures due to clipping
        for(Layer l : graph.getLayers()){
            if(l instanceof IOutputLayer){
                configureLossFnClippingIfPresent((IOutputLayer) l);
            }
        }

        for (int i = 0; i < inputs.length; i++)
            graph.setInput(i, inputs[i]);
        for (int i = 0; i < labels.length; i++)
            graph.setLabel(i, labels[i]);

        graph.setLayerMaskArrays(fMask, lMask);

        graph.computeGradientAndScore();
        Pair<Gradient, Double> gradAndScore = graph.gradientAndScore();

        ComputationGraphUpdater updater = new ComputationGraphUpdater(graph);
        updater.update(gradAndScore.getFirst(), 0, 0, graph.batchSize(), LayerWorkspaceMgr.noWorkspaces());

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
        MultiDataSet mds = new MultiDataSet(inputs, labels, fMask, lMask);
        INDArray params = graph.params(); //Assumption here: params is a view that we can modify in-place
        for (int i = 0; i < nParams; i++) {
            //Get param name
            if (i >= paramEnds[currParamNameIdx]) {
                currParamNameIdx++;
            }
            String paramName = paramNames.get(currParamNameIdx);
            if(excludeParams != null && excludeParams.contains(paramName)){
                log.info("Skipping parameters for parameter name: {}", paramName);
                i = paramEnds[currParamNameIdx++];
                continue;
            }

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
                                        + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus + ", paramValue = " + origValue);
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

        LayerWorkspaceMgr mgr = LayerWorkspaceMgr.noWorkspaces();

        //Basic sanity checks on input:
        if (epsilon <= 0.0 || epsilon > 0.1)
            throw new IllegalArgumentException("Invalid epsilon: expect epsilon in range (0,0.1], usually 1e-4 or so");
        if (maxRelError <= 0.0 || maxRelError > 0.25)
            throw new IllegalArgumentException("Invalid maxRelativeError: " + maxRelError);

        DataBuffer.Type dataType = DataTypeUtil.getDtypeFromContext();
        if (dataType != DataBuffer.Type.DOUBLE) {
            throw new IllegalStateException("Cannot perform gradient check: Datatype is not set to double precision ("
                            + "is: " + dataType + "). Double precision must be used for gradient checks. Set "
                            + "DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE); before using GradientCheckUtil");
        }

        //Check network configuration:
        layer.setInput(input, LayerWorkspaceMgr.noWorkspaces());
        Nd4j.getRandom().setSeed(rngSeed);
        layer.computeGradientAndScore(mgr);
        Pair<Gradient, Double> gradAndScore = layer.gradientAndScore();

        Updater updater = UpdaterCreator.getUpdater(layer);
        updater.update(layer, gradAndScore.getFirst(), 0, 0, layer.batchSize(), LayerWorkspaceMgr.noWorkspaces());

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
            layer.computeGradientAndScore(mgr);
            double scorePlus = layer.score();

            //(w-epsilon): Do forward pass and score
            params.putScalar(i, origValue - epsilon);
            Nd4j.getRandom().setSeed(rngSeed);
            layer.computeGradientAndScore(mgr);
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
                                        + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus + ", paramValue = " + origValue);
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
