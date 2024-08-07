/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.gradientcheck;

import lombok.*;
import lombok.experimental.Accessors;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.common.function.Consumer;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.common.primitives.Pair;
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
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
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

@Slf4j
public class GradientCheckUtil {


    private GradientCheckUtil() {}

    static {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }


    private static void configureLossFnClippingIfPresent(IOutputLayer outputLayer) {

        ILossFunction lfn = null;
        IActivation afn = null;
        if(outputLayer instanceof BaseOutputLayer) {
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

        log.info("Done setting clipping");
    }

    public enum PrintMode {
        ALL,
        ZEROS,
        FAILURES_ONLY
    }

    @Accessors(fluent = true)
    @Data
    @NoArgsConstructor
    public static class MLNConfig {
        private MultiLayerNetwork net;
        private INDArray input;
        private INDArray labels;
        private INDArray inputMask;
        private INDArray labelMask;
        private double epsilon = 1e-6;
        private double maxRelError = 1e-3;
        private double minAbsoluteError = 1e-8;
        private PrintMode print = PrintMode.ZEROS;
        private boolean exitOnFirstError = false;
        private boolean subset;
        private int maxPerParam;
        private Set<String> excludeParams;
        private Consumer<MultiLayerNetwork> callEachIter;
    }

    @Accessors(fluent = true)
    @Data
    @NoArgsConstructor
    public static class GraphConfig {
        private ComputationGraph net;
        private INDArray[] inputs;
        private INDArray[] labels;
        private INDArray[] inputMask;
        private INDArray[] labelMask;
        private double epsilon = 1e-6;
        private double maxRelError = 1e-3;
        private double minAbsoluteError = 1e-8;
        private PrintMode print = PrintMode.ZEROS;
        private boolean exitOnFirstError = false;
        private boolean subset;
        private int maxPerParam;
        private Set<String> excludeParams;
        private Consumer<ComputationGraph> callEachIter;
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
    @Deprecated
    public static boolean checkGradients(MultiLayerNetwork mln, double epsilon, double maxRelError,
                                         double minAbsoluteError, boolean print, boolean exitOnFirstError, INDArray input, INDArray labels) {
        return checkGradients(new MLNConfig().net(mln)
                .epsilon(epsilon)
                .maxRelError(maxRelError)
                .minAbsoluteError(minAbsoluteError)
                .print(PrintMode.FAILURES_ONLY)
                .exitOnFirstError(exitOnFirstError)
                .input(input)
                .labels(labels));
    }

    @Deprecated
    public static boolean checkGradients(MultiLayerNetwork mln, double epsilon, double maxRelError,
                                         double minAbsoluteError, boolean print, boolean exitOnFirstError,
                                         INDArray input, INDArray labels, INDArray inputMask, INDArray labelMask,
                                         boolean subset, int maxPerParam, Set<String> excludeParams, final Integer rngSeedResetEachIter) {
        Consumer<MultiLayerNetwork> c = null;
        if(rngSeedResetEachIter != null) {
            c = multiLayerNetwork -> Nd4j.getRandom().setSeed(rngSeedResetEachIter);
        }

        return checkGradients(new MLNConfig().net(mln).epsilon(epsilon).maxRelError(maxRelError).minAbsoluteError(minAbsoluteError).print(PrintMode.FAILURES_ONLY)
                .exitOnFirstError(exitOnFirstError).input(input).labels(labels).inputMask(inputMask).labelMask(labelMask).subset(subset).maxPerParam(maxPerParam).excludeParams(excludeParams).callEachIter(c));
    }

    public static boolean checkGradients(MLNConfig c) {
        //Basic sanity checks on input:
        if (c.epsilon <= 0.0 || c.epsilon > 0.1)
            throw new IllegalArgumentException("Invalid epsilon: expect epsilon in range (0,0.1], usually 1e-4 or so");
        if (c.maxRelError <= 0.0 || c.maxRelError > 0.25)
            throw new IllegalArgumentException("Invalid maxRelativeError: " + c.maxRelError);
        if (!(c.net.getOutputLayer() instanceof IOutputLayer))
            throw new IllegalArgumentException("Cannot check backprop gradients without OutputLayer");

        DataType dataType = DataTypeUtil.getDtypeFromContext();
        if (dataType != DataType.DOUBLE) {
            throw new IllegalStateException("Cannot perform gradient check: Datatype is not set to double precision ("
                    + "is: " + dataType + "). Double precision must be used for gradient checks. Set "
                    + "DataTypeUtil.setDTypeForContext(DataType.DOUBLE); before using GradientCheckUtil");
        }

        DataType netDataType = c.net.getLayerWiseConfigurations().getDataType();
        if (netDataType != DataType.DOUBLE) {
            throw new IllegalStateException("Cannot perform gradient check: Network datatype is not set to double precision ("
                    + "is: " + netDataType + "). Double precision must be used for gradient checks. Create network with .dataType(DataType.DOUBLE) before using GradientCheckUtil");
        }

        if(netDataType != c.net.params().dataType()) {
            throw new IllegalStateException("Parameters datatype does not match network configuration datatype ("
                    + "is: " + c.net.params().dataType() + "). If network datatype is set to DOUBLE, parameters must also be DOUBLE.");
        }


        //Check network configuration:
        int layerCount = 0;
        for (NeuralNetConfiguration n : c.net.getLayerWiseConfigurations().getConfs()) {
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


            }

            if (n.getLayer().getIDropout() != null && c.callEachIter == null) {
                throw new IllegalStateException("When gradient checking dropout, need to reset RNG seed each iter, or no" +
                        " dropout should be present during gradient checks - got dropout = "
                        + n.getLayer().getIDropout() + " for layer " + layerCount);
            }
        }

        //Set softmax clipping to 0 if necessary, to avoid spurious failures due to clipping
        for(Layer l : c.net.getLayers()) {
            if(l instanceof IOutputLayer) {
                configureLossFnClippingIfPresent((IOutputLayer) l);
            }
        }

        c.net.setInput(c.input);
        c.net.setLabels(c.labels);
        c.net.setLayerMaskArrays(c.inputMask, c.labelMask);
        if(c.callEachIter != null) {
            c.callEachIter.accept(c.net);
        }
        c.net.computeGradientAndScore();
        Pair<Gradient, Double> gradAndScore = c.net.gradientAndScore();

        Updater updater = c.net().createUpdater();
        updater.update(c.net, gradAndScore.getFirst(), 0, 0, c.net.batchSize(), LayerWorkspaceMgr.noWorkspaces());

        INDArray gradientToCheck = gradAndScore.getFirst().gradient().dup(); //need dup: gradients are a *view* of the full gradient array (which will change every time backprop is done)
        INDArray originalParams = c.net.params().dup(); //need dup: params are a *view* of full parameters

        val nParams = originalParams.length();

        Map<String, INDArray> paramTable = c.net.paramTable();
        List<String> paramNames = new ArrayList<>(paramTable.keySet());
        val paramEnds = new long[paramNames.size()];
        paramEnds[0] = paramTable.get(paramNames.get(0)).length();
        Map<String,Integer> stepSizeForParam;
        if(c.subset) {
            stepSizeForParam = new HashMap<>();
            stepSizeForParam.put(paramNames.get(0), (int) Math.max(1, paramTable.get(paramNames.get(0)).length() / c.maxPerParam));
        } else {
            stepSizeForParam = null;
        }
        for (int i = 1; i < paramEnds.length; i++) {
            val n = paramTable.get(paramNames.get(i)).length();
            paramEnds[i] = paramEnds[i - 1] + n;
            if(c.subset) {
                long ss = n / c.maxPerParam;
                if(ss == 0) {
                    ss = n;
                }

                if (ss > Integer.MAX_VALUE)
                    throw new ND4JArraySizeException();
                stepSizeForParam.put(paramNames.get(i), (int) ss);
            }
        }

        if(c.print == PrintMode.ALL) {
            int i = 0;
            for (Layer l : c.net.getLayers()) {
                Set<String> s = l.paramTable().keySet();
                log.info("Layer " + i + ": " + l.getClass().getSimpleName() + " - params " + s);
                i++;
            }
        }


        int totalNFailures = 0;
        double maxError = 0.0;
        DataSet ds = new DataSet(c.input, c.labels, c.inputMask, c.labelMask);
        int currParamNameIdx = 0;

        if(c.excludeParams != null && !c.excludeParams.isEmpty()) {
            log.info("NOTE: parameters will be skipped due to config: {}", c.excludeParams);
        }

        INDArray params = c.net.params(); //Assumption here: params is a view that we can modify in-place
        for (long i = 0; i < nParams;) {
            //Get param name
            if (i >= paramEnds[currParamNameIdx]) {
                currParamNameIdx++;
            }
            String paramName = paramNames.get(currParamNameIdx);
            if(c.excludeParams != null && c.excludeParams.contains(paramName)) {
                i = paramEnds[currParamNameIdx++];
                continue;
            }

            //(w+epsilon): Do forward pass and score
            double origValue = params.getDouble(i);
            params.putScalar(i, origValue + c.epsilon);
            if(c.callEachIter != null) {
                c.callEachIter.accept(c.net);
            }
            double scorePlus = c.net.score(ds, true);

            //(w-epsilon): Do forward pass and score
            params.putScalar(i, origValue - c.epsilon);
            if(c.callEachIter != null) {
                c.callEachIter.accept(c.net);
            }
            double scoreMinus = c.net.score(ds, true);

            //Reset original param value
            params.putScalar(i, origValue);

            //Calculate numerical parameter gradient:
            double scoreDelta = scorePlus - scoreMinus;

            double numericalGradient = scoreDelta / (2 * c.epsilon);
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
            if (relError > c.maxRelError || Double.isNaN(relError)) {
                double absError = Math.abs(backpropGradient - numericalGradient);
                if (absError < c.minAbsoluteError) {
                    if(c.print == PrintMode.ALL || c.print == PrintMode.ZEROS && absError == 0.0) {
                        log.info("MLN Param " + i + " (" + paramName + ") passed: grad= " + backpropGradient
                                + ", numericalGrad= " + numericalGradient + ", relError= " + relError
                                + "; absolute error = " + absError + " < minAbsoluteError = " + c.minAbsoluteError);
                    }
                } else {
                    log.info("MLN Param " + i + " (" + paramName + ") FAILED: grad= " + backpropGradient
                            + ", numericalGrad= " + numericalGradient + ", relError= " + relError
                            + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus + ", paramValue = " + origValue);
                    if (c.exitOnFirstError)
                        return false;
                    totalNFailures++;
                }
            } else if (c.print == PrintMode.ALL) {
                log.info("Param " + i + " (" + paramName + ") passed: grad= " + backpropGradient + ", numericalGrad= "
                        + numericalGradient + ", relError= " + relError);
            }

            long step;
            if(c.subset) {
                step = stepSizeForParam.get(paramName);
                if(i + step > paramEnds[currParamNameIdx] + 1) {
                    step = paramEnds[currParamNameIdx]+1 - i;
                }
            } else {
                step = 1;
            }

            i += step;
        }

        val nPass = nParams - totalNFailures;
        log.info("GradientCheckUtil.checkGradients(): " + nParams + " params checked, " + nPass + " passed, "
                + totalNFailures + " failed. Largest relative error = " + maxError);

        return totalNFailures == 0;
    }

    public static boolean checkGradients(GraphConfig c) {
        //Basic sanity checks on input:
        if (c.epsilon <= 0.0 || c.epsilon > 0.1)
            throw new IllegalArgumentException("Invalid epsilon: expect epsilon in range (0,0.1], usually 1e-4 or so");
        if (c.maxRelError <= 0.0 || c.maxRelError > 0.25)
            throw new IllegalArgumentException("Invalid maxRelativeError: " + c.maxRelError);

        if (c.net.getNumInputArrays() != c.inputs.length)
            throw new IllegalArgumentException("Invalid input arrays: expect " + c.net.getNumInputArrays() + " inputs");
        if (c.net.getNumOutputArrays() != c.labels.length)
            throw new IllegalArgumentException(
                    "Invalid labels arrays: expect " + c.net.getNumOutputArrays() + " outputs");

        DataType dataType = DataTypeUtil.getDtypeFromContext();
        if (dataType != DataType.DOUBLE) {
            throw new IllegalStateException("Cannot perform gradient check: Datatype is not set to double precision ("
                    + "is: " + dataType + "). Double precision must be used for gradient checks. Set "
                    + "DataTypeUtil.setDTypeForContext(DataType.DOUBLE); before using GradientCheckUtil");
        }

        DataType netDataType = c.net.getConfiguration().getDataType();
        if (netDataType != DataType.DOUBLE) {
            throw new IllegalStateException("Cannot perform gradient check: Network datatype is not set to double precision ("
                    + "is: " + netDataType + "). Double precision must be used for gradient checks. Create network with .dataType(DataType.DOUBLE) before using GradientCheckUtil");
        }

        if(netDataType != c.net.params().dataType()) {
            throw new IllegalStateException("Parameters datatype does not match network configuration datatype ("
                    + "is: " + c.net.params().dataType() + "). If network datatype is set to DOUBLE, parameters must also be DOUBLE.");
        }

        //Check configuration
        int layerCount = 0;
        for (String vertexName : c.net.getConfiguration().getVertices().keySet()) {
            GraphVertex gv = c.net.getConfiguration().getVertices().get(vertexName);
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


            }

            if (lv.getLayerConf().getLayer().getIDropout() != null && c.callEachIter == null) {
                throw new IllegalStateException("When gradient checking dropout, rng seed must be reset each iteration, or no" +
                        " dropout should be present during gradient checks - got dropout = "
                        + lv.getLayerConf().getLayer().getIDropout() + " for layer " + layerCount);
            }
        }

        //Set softmax clipping to 0 if necessary, to avoid spurious failures due to clipping
        for(Layer l : c.net.getLayers()) {
            if(l instanceof IOutputLayer) {
                configureLossFnClippingIfPresent((IOutputLayer) l);
            }
        }

        for (int i = 0; i < c.inputs.length; i++)
            c.net.setInput(i, c.inputs[i]);
        for (int i = 0; i < c.labels.length; i++)
            c.net.setLabel(i, c.labels[i]);

        c.net.setLayerMaskArrays(c.inputMask, c.labelMask);

        if(c.callEachIter != null){
            c.callEachIter.accept(c.net);
        }

        c.net.computeGradientAndScore();
        Pair<Gradient, Double> gradAndScore = c.net.gradientAndScore();

        ComputationGraphUpdater updater = new ComputationGraphUpdater(c.net);
        updater.update(gradAndScore.getFirst(), 0, 0, c.net.batchSize(), LayerWorkspaceMgr.noWorkspaces());

        INDArray gradientToCheck = gradAndScore.getFirst().gradient().dup(); //need dup: gradients are a *view* of the full gradient array (which will change every time backprop is done)
        INDArray originalParams = c.net.params().dup(); //need dup: params are a *view* of full parameters

        val nParams = originalParams.length();

        Map<String, INDArray> paramTable = c.net.paramTable();
        List<String> paramNames = new ArrayList<>(paramTable.keySet());
        val paramEnds = new long[paramNames.size()];
        paramEnds[0] = paramTable.get(paramNames.get(0)).length();
        for (int i = 1; i < paramEnds.length; i++) {
            paramEnds[i] = paramEnds[i - 1] + paramTable.get(paramNames.get(i)).length();
        }

        if(c.excludeParams != null && !c.excludeParams.isEmpty()){
            log.info("NOTE: parameters will be skipped due to config: {}", c.excludeParams);
        }

        int currParamNameIdx = 0;
        int totalNFailures = 0;
        double maxError = 0.0;
        MultiDataSet mds = new MultiDataSet(c.inputs, c.labels, c.inputMask, c.labelMask);
        INDArray params = c.net.params(); //Assumption here: params is a view that we can modify in-place
        for (long i = 0; i < nParams; i++) {
            //Get param name
            if (i >= paramEnds[currParamNameIdx]) {
                currParamNameIdx++;
            }
            String paramName = paramNames.get(currParamNameIdx);
            if(c.excludeParams != null && c.excludeParams.contains(paramName)){
                //log.info("Skipping parameters for parameter name: {}", paramName);
                i = paramEnds[currParamNameIdx++];
                continue;
            }

            //(w+epsilon): Do forward pass and score
            double origValue = params.getDouble(i);

            params.putScalar(i, origValue + c.epsilon);
            if(c.callEachIter != null) {
                c.callEachIter.accept(c.net);
            }
            double scorePlus = c.net.score(mds, true); //training == true for batch norm, etc (scores and gradients need to be calculated on same thing)

            //(w-epsilon): Do forward pass and score
            params.putScalar(i, origValue - c.epsilon);
            if(c.callEachIter != null) {
                c.callEachIter.accept(c.net);
            }
            double scoreMinus = c.net.score(mds, true);

            //Reset original param value
            params.putScalar(i, origValue);

            //Calculate numerical parameter gradient:
            double scoreDelta = scorePlus - scoreMinus;

            double numericalGradient = scoreDelta / (2 * c.epsilon);
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
            if (relError > c.maxRelError || Double.isNaN(relError)) {
                double absError = Math.abs(backpropGradient - numericalGradient);
                if (absError < c.minAbsoluteError) {
                    if(c.print == PrintMode.ALL || c.print == PrintMode.ZEROS && absError == 0.0) {
                        log.info("Param " + i + " (" + paramName + ") passed: grad= " + backpropGradient
                                + ", numericalGrad= " + numericalGradient + ", relError= " + relError
                                + "; absolute error = " + absError + " < minAbsoluteError = " + c.minAbsoluteError);
                    }
                } else {
                    log.info("Param " + i + " (" + paramName + ") FAILED: grad= " + backpropGradient
                            + ", numericalGrad= " + numericalGradient + ", relError= " + relError
                            + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus + ", paramValue = " + origValue);
                    if (c.exitOnFirstError)
                        return false;
                    totalNFailures++;
                }
            } else if (c.print == PrintMode.ALL) {
                log.info("Param " + i + " (" + paramName + ") passed: grad= " + backpropGradient + ", numericalGrad= "
                        + numericalGradient + ", relError= " + relError);
            }
        }

        val nPass = nParams - totalNFailures;
        log.info("GradientCheckUtil.checkGradients(): " + nParams + " params checked, " + nPass + " passed, "
                + totalNFailures + " failed. Largest relative error = " + maxError);

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

        DataType dataType = DataTypeUtil.getDtypeFromContext();
        if (dataType != DataType.DOUBLE) {
            throw new IllegalStateException("Cannot perform gradient check: Datatype is not set to double precision ("
                    + "is: " + dataType + "). Double precision must be used for gradient checks. Set "
                    + "DataTypeUtil.setDTypeForContext(DataType.DOUBLE); before using GradientCheckUtil");
        }

        //Check network configuration:
        layer.setInput(input, LayerWorkspaceMgr.noWorkspaces());
        Nd4j.getRandom().setSeed(rngSeed);
        layer.computeGradientAndScore(mgr);
        Pair<Gradient, Double> gradAndScore = layer.gradientAndScore();

        Updater updater = layer.createUpdater();
        updater.update(layer, gradAndScore.getFirst(), 0, 0, layer.batchSize(), LayerWorkspaceMgr.noWorkspaces());

        INDArray gradientToCheck = gradAndScore.getFirst().gradient().dup(); //need dup: gradients are a *view* of the full gradient array (which will change every time backprop is done)
        INDArray originalParams = layer.params().dup(); //need dup: params are a *view* of full parameters

        val nParams = originalParams.length();

        Map<String, INDArray> paramTable = layer.paramTable();
        List<String> paramNames = new ArrayList<>(paramTable.keySet());
        val paramEnds = new long[paramNames.size()];
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
            val nPass = nParams - totalNFailures;
            log.info("GradientCheckUtil.checkGradients(): " + nParams + " params checked, " + nPass + " passed, "
                    + totalNFailures + " failed. Largest relative error = " + maxError);
        }

        return totalNFailures == 0;
    }
}
