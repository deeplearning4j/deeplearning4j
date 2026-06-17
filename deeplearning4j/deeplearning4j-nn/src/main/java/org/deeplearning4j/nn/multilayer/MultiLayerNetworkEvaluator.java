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

package org.deeplearning4j.nn.multilayer;

import lombok.NonNull;
import lombok.val;
import org.deeplearning4j.datasets.iterator.MultiDataSetWrapperIterator;
import org.deeplearning4j.nn.api.FwdPassType;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.util.CrashReportingUtil;
import org.deeplearning4j.util.OutputLayerUtil;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROC;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.abstracts.DummyWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.AsyncDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.List;

/**
 * Helper class containing all evaluation-related methods extracted from {@link MultiLayerNetwork}.
 * Each method delegates to the corresponding implementation here; the public API on
 * {@link MultiLayerNetwork} is preserved unchanged.
 */
public class MultiLayerNetworkEvaluator {

    private final MultiLayerNetwork network;

    public MultiLayerNetworkEvaluator(MultiLayerNetwork network) {
        this.network = network;
    }

    /**
     * See {@link MultiLayerNetwork#evaluate(DataSetIterator)}.
     */
    @SuppressWarnings("unchecked")
    public <T extends Evaluation> T evaluate(@NonNull DataSetIterator iterator) {
        return (T) network.evaluate(iterator, null);
    }

    /**
     * See {@link MultiLayerNetwork#evaluate(MultiDataSetIterator)}.
     */
    public Evaluation evaluate(@NonNull MultiDataSetIterator iterator) {
        return network.evaluate(new MultiDataSetWrapperIterator(iterator));
    }

    /**
     * See {@link MultiLayerNetwork#evaluateRegression(DataSetIterator)}.
     */
    @SuppressWarnings("unchecked")
    public <T extends RegressionEvaluation> T evaluateRegression(DataSetIterator iterator) {
        return (T) network.doEvaluation(iterator,
                new RegressionEvaluation(iterator.totalOutcomes()))[0];
    }

    /**
     * See {@link MultiLayerNetwork#evaluateRegression(MultiDataSetIterator)}.
     */
    public RegressionEvaluation evaluateRegression(MultiDataSetIterator iterator) {
        return evaluateRegression(new MultiDataSetWrapperIterator(iterator));
    }

    /**
     * See {@link MultiLayerNetwork#evaluateROC(DataSetIterator)}.
     *
     * @deprecated use {@link #evaluateROC(DataSetIterator, int)}
     */
    @Deprecated
    @SuppressWarnings("unchecked")
    public <T extends ROC> T evaluateROC(DataSetIterator iterator) {
        return evaluateROC(iterator, 0);
    }

    /**
     * See {@link MultiLayerNetwork#evaluateROC(DataSetIterator, int)}.
     */
    @SuppressWarnings("unchecked")
    public <T extends ROC> T evaluateROC(DataSetIterator iterator, int rocThresholdSteps) {
        Layer outputLayer = network.getOutputLayer();
        if (network.getLayerWiseConfigurations().isValidateOutputLayerConfig()) {
            OutputLayerUtil.validateOutputLayerForClassifierEvaluation(
                    outputLayer.conf().getLayer(), ROC.class);
        }
        return (T) network.doEvaluation(iterator,
                new org.nd4j.evaluation.classification.ROC(rocThresholdSteps))[0];
    }

    /**
     * See {@link MultiLayerNetwork#evaluateROCMultiClass(DataSetIterator)}.
     *
     * @deprecated use {@link #evaluateROCMultiClass(DataSetIterator, int)}
     */
    @Deprecated
    @SuppressWarnings("unchecked")
    public <T extends ROCMultiClass> T evaluateROCMultiClass(DataSetIterator iterator) {
        return evaluateROCMultiClass(iterator, 0);
    }

    /**
     * See {@link MultiLayerNetwork#evaluateROCMultiClass(DataSetIterator, int)}.
     */
    @SuppressWarnings("unchecked")
    public <T extends ROCMultiClass> T evaluateROCMultiClass(DataSetIterator iterator,
            int rocThresholdSteps) {
        Layer outputLayer = network.getOutputLayer();
        if (network.getLayerWiseConfigurations().isValidateOutputLayerConfig()) {
            OutputLayerUtil.validateOutputLayerForClassifierEvaluation(
                    outputLayer.conf().getLayer(), ROCMultiClass.class);
        }
        return (T) network.doEvaluation(iterator,
                new org.nd4j.evaluation.classification.ROCMultiClass(rocThresholdSteps))[0];
    }

    /**
     * See {@link MultiLayerNetwork#doEvaluation(DataSetIterator, IEvaluation[])}.
     */
    public <T extends IEvaluation> T[] doEvaluation(DataSetIterator iterator, T... evaluations) {
        try {
            return doEvaluationHelper(iterator, evaluations);
        } catch (OutOfMemoryError e) {
            CrashReportingUtil.writeMemoryCrashDump(network, e);
            throw e;
        }
    }

    /**
     * See {@link MultiLayerNetwork#doEvaluation(MultiDataSetIterator, IEvaluation[])}.
     */
    public <T extends IEvaluation> T[] doEvaluation(MultiDataSetIterator iterator,
            T[] evaluations) {
        return doEvaluation(new MultiDataSetWrapperIterator(iterator), evaluations);
    }

    /**
     * See {@link MultiLayerNetwork#doEvaluationHelper(DataSetIterator, IEvaluation[])}.
     */
    public <T extends IEvaluation> T[] doEvaluationHelper(DataSetIterator iterator,
            T... evaluations) {
        if (!iterator.hasNext() && iterator.resetSupported()) {
            iterator.reset();
        }

        DataSetIterator iter = iterator.asyncSupported()
                ? new AsyncDataSetIterator(iterator, 2, true) : iterator;

        WorkspaceMode cMode = network.layerWiseConfigurations.getTrainingWorkspaceMode();
        network.layerWiseConfigurations.setTrainingWorkspaceMode(
                network.layerWiseConfigurations.getInferenceWorkspaceMode());

        boolean useRnnSegments = (network.layerWiseConfigurations.getBackpropType()
                == BackpropType.TruncatedBPTT);

        MemoryWorkspace outputWs;
        if (network.getLayerWiseConfigurations().getInferenceWorkspaceMode()
                == WorkspaceMode.ENABLED) {
            outputWs = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                    MultiLayerNetwork.WS_ALL_LAYERS_ACT_CONFIG,
                    MultiLayerNetwork.WS_OUTPUT_MEM);
        } else {
            outputWs = new DummyWorkspace();
        }

        while (iter.hasNext()) {
            DataSet next = iter.next();

            if (next.getFeatures() == null || next.getLabels() == null)
                continue;

            INDArray features = next.getFeatures();
            INDArray labels = next.getLabels();
            INDArray fMask = next.getFeaturesMaskArray();
            INDArray lMask = next.getLabelsMaskArray();
            List<Serializable> meta = next.getExampleMetaData();

            if (!useRnnSegments) {
                try (MemoryWorkspace ws = outputWs.notifyScopeEntered()) {
                    Layer[] layers = network.getLayers();
                    INDArray out = network.outputOfLayerDetached(false, FwdPassType.STANDARD,
                            layers.length - 1, features, fMask, lMask, ws);

                    try (MemoryWorkspace wsO = Nd4j.getWorkspaceManager()
                            .scopeOutOfWorkspaces()) {
                        for (T evaluation : evaluations)
                            evaluation.eval(labels, out, lMask, meta);
                    }
                }
            } else {
                network.rnnClearPreviousState();

                val fwdLen = network.layerWiseConfigurations.getTbpttFwdLength();
                val tsLength = features.size(2);
                long nSubsets = tsLength / fwdLen;
                if (tsLength % fwdLen != 0)
                    nSubsets++;

                for (int i = 0; i < nSubsets; i++) {
                    val startTimeIdx = i * fwdLen;
                    val endTimeIdx = Math.min(startTimeIdx + fwdLen, tsLength);

                    if (endTimeIdx > Integer.MAX_VALUE)
                        throw new ND4JArraySizeException();
                    INDArray[] subsets = network.getSubsetsForTbptt(startTimeIdx,
                            (int) endTimeIdx, features, labels, fMask, lMask);

                    network.setLayerMaskArrays(subsets[2], subsets[3]);

                    try (MemoryWorkspace ws = outputWs.notifyScopeEntered()) {
                        INDArray outSub = network.rnnTimeStep(subsets[0], ws);
                        try (MemoryWorkspace wsO = Nd4j.getWorkspaceManager()
                                .scopeOutOfWorkspaces()) {
                            for (T evaluation : evaluations)
                                evaluation.eval(subsets[1], outSub, subsets[3]);
                        }
                    }
                }
            }

            network.clearLayersStates();
        }

        if (iterator.asyncSupported())
            ((AsyncDataSetIterator) iter).shutdown();

        network.layerWiseConfigurations.setTrainingWorkspaceMode(cMode);

        return evaluations;
    }

    /**
     * See {@link MultiLayerNetwork#evaluate(DataSetIterator, List)}.
     */
    public Evaluation evaluate(DataSetIterator iterator, List<String> labelsList) {
        return evaluate(iterator, labelsList, 1);
    }

    /**
     * See {@link MultiLayerNetwork#evaluate(DataSetIterator, List, int)}.
     */
    public Evaluation evaluate(DataSetIterator iterator, List<String> labelsList, int topN) {
        Layer[] layers = network.getLayers();
        if (layers == null || !(network.getOutputLayer() instanceof IOutputLayer)) {
            throw new IllegalStateException("Cannot evaluate network with no output layer");
        }
        if (labelsList == null) {
            try {
                labelsList = iterator.getLabels();
            } catch (Throwable t) {
                // Ignore, maybe UnsupportedOperationException etc
            }
        }

        Layer outputLayer = network.getOutputLayer();
        if (network.getLayerWiseConfigurations().isValidateOutputLayerConfig()) {
            OutputLayerUtil.validateOutputLayerForClassifierEvaluation(
                    outputLayer.conf().getLayer(), Evaluation.class);
        }

        Evaluation e = new org.nd4j.evaluation.classification.Evaluation(labelsList, topN);
        network.doEvaluation(iterator, e);

        return e;
    }
}
