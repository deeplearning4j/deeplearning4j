/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.dataset.api.preprocessor;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerType;
import org.nd4j.linalg.dataset.api.preprocessor.stats.NormalizerStats;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Pre processor for MultiDataSet that can be configured to use different normalization strategies for different inputs
 * and outputs, or none at all. Can be used for example when one input should be normalized, but a different one should
 * be untouched because it's the input for an embedding layer. Alternatively, one might want to mix standardization and
 * min-max scaling for different inputs and outputs.
 * <p>
 * By default, no normalization is applied. There are methods to configure the desired normalization strategy for inputs
 * and outputs either globally or on an individual input/output level. Specific input/output strategies will override
 * global ones.
 *
 * @author Ede Meijer
 */
@EqualsAndHashCode(callSuper = false)
@Setter
public class MultiNormalizerHybrid extends AbstractNormalizer implements MultiDataNormalization, Serializable {
    private Map<Integer, NormalizerStats> inputStats;
    private Map<Integer, NormalizerStats> outputStats;
    @Getter
    private NormalizerStrategy globalInputStrategy;
    @Getter
    private NormalizerStrategy globalOutputStrategy;
    @Getter
    private Map<Integer, NormalizerStrategy> perInputStrategies = new HashMap<>();
    @Getter
    private Map<Integer, NormalizerStrategy> perOutputStrategies = new HashMap<>();

    /**
     * Apply standardization to all inputs, except the ones individually configured
     *
     * @return the normalizer
     */
    public MultiNormalizerHybrid standardizeAllInputs() {
        globalInputStrategy = new StandardizeStrategy();
        return this;
    }

    /**
     * Apply min-max scaling to all inputs, except the ones individually configured
     *
     * @return the normalizer
     */
    public MultiNormalizerHybrid minMaxScaleAllInputs() {
        globalInputStrategy = new MinMaxStrategy();
        return this;
    }

    /**
     * Apply min-max scaling to all inputs, except the ones individually configured
     *
     * @param rangeFrom lower bound of the target range
     * @param rangeTo   upper bound of the target range
     * @return the normalizer
     */
    public MultiNormalizerHybrid minMaxScaleAllInputs(double rangeFrom, double rangeTo) {
        globalInputStrategy = new MinMaxStrategy(rangeFrom, rangeTo);
        return this;
    }

    /**
     * Apply standardization to a specific input, overriding the global input strategy if any
     *
     * @param input the index of the input
     * @return the normalizer
     */
    public MultiNormalizerHybrid standardizeInput(int input) {
        perInputStrategies.put(input, new StandardizeStrategy());
        return this;
    }

    /**
     * Apply min-max scaling to a specific input, overriding the global input strategy if any
     *
     * @param input the index of the input
     * @return the normalizer
     */
    public MultiNormalizerHybrid minMaxScaleInput(int input) {
        perInputStrategies.put(input, new MinMaxStrategy());
        return this;
    }

    /**
     * Apply min-max scaling to a specific input, overriding the global input strategy if any
     *
     * @param input     the index of the input
     * @param rangeFrom lower bound of the target range
     * @param rangeTo   upper bound of the target range
     * @return the normalizer
     */
    public MultiNormalizerHybrid minMaxScaleInput(int input, double rangeFrom, double rangeTo) {
        perInputStrategies.put(input, new MinMaxStrategy(rangeFrom, rangeTo));
        return this;
    }

    /**
     * Apply standardization to all outputs, except the ones individually configured
     *
     * @return the normalizer
     */
    public MultiNormalizerHybrid standardizeAllOutputs() {
        globalOutputStrategy = new StandardizeStrategy();
        return this;
    }

    /**
     * Apply min-max scaling to all outputs, except the ones individually configured
     *
     * @return the normalizer
     */
    public MultiNormalizerHybrid minMaxScaleAllOutputs() {
        globalOutputStrategy = new MinMaxStrategy();
        return this;
    }

    /**
     * Apply min-max scaling to all outputs, except the ones individually configured
     *
     * @param rangeFrom lower bound of the target range
     * @param rangeTo   upper bound of the target range
     * @return the normalizer
     */
    public MultiNormalizerHybrid minMaxScaleAllOutputs(double rangeFrom, double rangeTo) {
        globalOutputStrategy = new MinMaxStrategy(rangeFrom, rangeTo);
        return this;
    }

    /**
     * Apply standardization to a specific output, overriding the global output strategy if any
     *
     * @param output the index of the input
     * @return the normalizer
     */
    public MultiNormalizerHybrid standardizeOutput(int output) {
        perOutputStrategies.put(output, new StandardizeStrategy());
        return this;
    }

    /**
     * Apply min-max scaling to a specific output, overriding the global output strategy if any
     *
     * @param output the index of the input
     * @return the normalizer
     */
    public MultiNormalizerHybrid minMaxScaleOutput(int output) {
        perOutputStrategies.put(output, new MinMaxStrategy());
        return this;
    }

    /**
     * Apply min-max scaling to a specific output, overriding the global output strategy if any
     *
     * @param output    the index of the input
     * @param rangeFrom lower bound of the target range
     * @param rangeTo   upper bound of the target range
     * @return the normalizer
     */
    public MultiNormalizerHybrid minMaxScaleOutput(int output, double rangeFrom, double rangeTo) {
        perOutputStrategies.put(output, new MinMaxStrategy(rangeFrom, rangeTo));
        return this;
    }

    /**
     * Get normalization statistics for a given input.
     *
     * @param input the index of the input
     * @return implementation of NormalizerStats corresponding to the normalization strategy selected
     */
    public NormalizerStats getInputStats(int input) {
        return getInputStats().get(input);
    }

    /**
     * Get normalization statistics for a given output.
     *
     * @param output the index of the output
     * @return implementation of NormalizerStats corresponding to the normalization strategy selected
     */
    public NormalizerStats getOutputStats(int output) {
        return getOutputStats().get(output);
    }

    /**
     * Get the map of normalization statistics per input
     * 
     * @return map of input indices pointing to NormalizerStats instances
     */
    public Map<Integer, NormalizerStats> getInputStats() {
        assertIsFit();
        return inputStats;
    }

    /**
     * Get the map of normalization statistics per output
     *
     * @return map of output indices pointing to NormalizerStats instances
     */
    public Map<Integer, NormalizerStats> getOutputStats() {
        assertIsFit();
        return outputStats;
    }

    /**
     * Fit a MultiDataSet (only compute based on the statistics from this dataset)
     *
     * @param dataSet the dataset to compute on
     */
    @Override
    public void fit(@NonNull MultiDataSet dataSet) {
        Map<Integer, NormalizerStats.Builder> inputStatsBuilders = new HashMap<>();
        Map<Integer, NormalizerStats.Builder> outputStatsBuilders = new HashMap<>();

        fitPartial(dataSet, inputStatsBuilders, outputStatsBuilders);

        inputStats = buildAllStats(inputStatsBuilders);
        outputStats = buildAllStats(outputStatsBuilders);
    }

    /**
     * Iterates over a dataset
     * accumulating statistics for normalization
     *
     * @param iterator the iterator to use for collecting statistics
     */
    @Override
    public void fit(@NonNull MultiDataSetIterator iterator) {
        Map<Integer, NormalizerStats.Builder> inputStatsBuilders = new HashMap<>();
        Map<Integer, NormalizerStats.Builder> outputStatsBuilders = new HashMap<>();

        iterator.reset();
        while (iterator.hasNext()) {
            fitPartial(iterator.next(), inputStatsBuilders, outputStatsBuilders);
        }

        inputStats = buildAllStats(inputStatsBuilders);
        outputStats = buildAllStats(outputStatsBuilders);
    }

    private void fitPartial(MultiDataSet dataSet, Map<Integer, NormalizerStats.Builder> inputStatsBuilders,
                    Map<Integer, NormalizerStats.Builder> outputStatsBuilders) {
        ensureStatsBuilders(inputStatsBuilders, globalInputStrategy, perInputStrategies, dataSet.numFeatureArrays());
        ensureStatsBuilders(outputStatsBuilders, globalOutputStrategy, perOutputStrategies, dataSet.numLabelsArrays());

        for (int index : inputStatsBuilders.keySet()) {
            inputStatsBuilders.get(index).add(dataSet.getFeatures(index), dataSet.getFeaturesMaskArray(index));
        }
        for (int index : outputStatsBuilders.keySet()) {
            outputStatsBuilders.get(index).add(dataSet.getLabels(index), dataSet.getLabelsMaskArray(index));
        }
    }

    private void ensureStatsBuilders(Map<Integer, NormalizerStats.Builder> builders, NormalizerStrategy globalStrategy,
                    Map<Integer, NormalizerStrategy> perArrayStrategies, int numArrays) {
        if (builders.isEmpty()) {
            for (int i = 0; i < numArrays; i++) {
                NormalizerStrategy strategy = getStrategy(globalStrategy, perArrayStrategies, i);
                if (strategy != null) {
                    builders.put(i, strategy.newStatsBuilder());
                }
            }
        }
    }

    private Map<Integer, NormalizerStats> buildAllStats(@NonNull Map<Integer, NormalizerStats.Builder> builders) {
        Map<Integer, NormalizerStats> result = new HashMap<>(builders.size());
        for (int index : builders.keySet()) {
            result.put(index, builders.get(index).build());
        }
        return result;
    }

    /**
     * Transform the dataset
     *
     * @param data the dataset to pre process
     */
    @Override
    public void transform(@NonNull MultiDataSet data) {
        preProcess(data);
    }

    @Override
    public void preProcess(@NonNull MultiDataSet data) {
        preProcess(data.getFeatures(), data.getFeaturesMaskArrays(), globalInputStrategy, perInputStrategies,
                        getInputStats());
        preProcess(data.getLabels(), data.getLabelsMaskArrays(), globalOutputStrategy, perOutputStrategies,
                        getOutputStats());
    }

    private void preProcess(INDArray[] arrays, INDArray[] masks, NormalizerStrategy globalStrategy,
                    Map<Integer, NormalizerStrategy> perArrayStrategy, Map<Integer, NormalizerStats> stats) {

        if (arrays != null) {
            for (int i = 0; i < arrays.length; i++) {
                NormalizerStrategy strategy = getStrategy(globalStrategy, perArrayStrategy, i);
                if (strategy != null) {
                    //noinspection unchecked
                    strategy.preProcess(arrays[i], masks == null ? null : masks[i], stats.get(i));
                }
            }
        }
    }

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance (arrays are modified in-place)
     *
     * @param data MultiDataSet to revert the normalization on
     */
    @Override
    public void revert(@NonNull MultiDataSet data) {
        revertFeatures(data.getFeatures(), data.getFeaturesMaskArrays());
        revertLabels(data.getLabels(), data.getLabelsMaskArrays());
    }

    @Override
    public NormalizerType getType() {
        return NormalizerType.MULTI_HYBRID;
    }

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the entire inputs array
     *
     * @param features The normalized array of inputs
     */
    @Override
    public void revertFeatures(@NonNull INDArray[] features) {
        revertFeatures(features, null);
    }

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the entire inputs array
     *
     * @param features   The normalized array of inputs
     * @param maskArrays Optional mask arrays belonging to the inputs
     */
    @Override
    public void revertFeatures(@NonNull INDArray[] features, INDArray[] maskArrays) {
        for (int i = 0; i < features.length; i++) {
            revertFeatures(features, maskArrays, i);
        }
    }

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the features of a particular input
     *
     * @param features   The normalized array of inputs
     * @param maskArrays Optional mask arrays belonging to the inputs
     * @param input      the index of the input to revert normalization on
     */
    public void revertFeatures(@NonNull INDArray[] features, INDArray[] maskArrays, int input) {
        NormalizerStrategy strategy = getStrategy(globalInputStrategy, perInputStrategies, input);
        if (strategy != null) {
            INDArray mask = (maskArrays == null ? null : maskArrays[input]);
            //noinspection unchecked
            strategy.revert(features[input], mask, getInputStats(input));
        }
    }

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the entire outputs array
     *
     * @param labels The normalized array of outputs
     */
    @Override
    public void revertLabels(@NonNull INDArray[] labels) {
        revertLabels(labels, null);
    }

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the entire outputs array
     *
     * @param labels     The normalized array of outputs
     * @param maskArrays Optional mask arrays belonging to the outputs
     */
    @Override
    public void revertLabels(@NonNull INDArray[] labels, INDArray[] maskArrays) {
        for (int i = 0; i < labels.length; i++) {
            revertLabels(labels, maskArrays, i);
        }
    }

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the labels of a particular output
     *
     * @param labels     The normalized array of outputs
     * @param maskArrays Optional mask arrays belonging to the outputs
     * @param output     the index of the output to revert normalization on
     */
    public void revertLabels(@NonNull INDArray[] labels, INDArray[] maskArrays, int output) {
        NormalizerStrategy strategy = getStrategy(globalOutputStrategy, perOutputStrategies, output);
        if (strategy != null) {
            INDArray mask = (maskArrays == null ? null : maskArrays[output]);
            //noinspection unchecked
            strategy.revert(labels[output], mask, getOutputStats(output));
        }
    }

    private NormalizerStrategy getStrategy(NormalizerStrategy globalStrategy,
                    Map<Integer, NormalizerStrategy> perArrayStrategy, int index) {
        NormalizerStrategy strategy = globalStrategy;
        if (perArrayStrategy.containsKey(index)) {
            strategy = perArrayStrategy.get(index);
        }
        return strategy;
    }

    @Override
    protected boolean isFit() {
        return inputStats != null;
    }
}
