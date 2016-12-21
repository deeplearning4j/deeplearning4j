package org.nd4j.linalg.dataset.api.preprocessor;

import lombok.EqualsAndHashCode;
import lombok.NonNull;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.stats.NormalizerStats;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

@EqualsAndHashCode
public class HybridMultiDataSetNormalizer implements MultiDataNormalization, Serializable {
    @Setter private Map<Integer, NormalizerStats> inputStats;
    @Setter private Map<Integer, NormalizerStats> outputStats;
    private NormalizerStrategy globalInputStrategy;
    private NormalizerStrategy globalOutputStrategy;
    private Map<Integer, NormalizerStrategy> perInputStrategies = new HashMap<>();
    private Map<Integer, NormalizerStrategy> perOutputStrategies = new HashMap<>();

    public HybridMultiDataSetNormalizer standardizeAllInputs() {
        globalInputStrategy = new StandardizeStrategy();
        return this;
    }

    public HybridMultiDataSetNormalizer minMaxScaleAllInputs() {
        globalOutputStrategy = new MinMaxStrategy();
        return this;
    }

    public HybridMultiDataSetNormalizer minMaxScaleAllInputs(double rangeFrom, double rangeTo) {
        globalOutputStrategy = new MinMaxStrategy(rangeFrom, rangeTo);
        return this;
    }

    public HybridMultiDataSetNormalizer standardizeInput(int input) {
        perInputStrategies.put(input, new StandardizeStrategy());
        return this;
    }

    public HybridMultiDataSetNormalizer minMaxScaleInput(int input) {
        perInputStrategies.put(input, new MinMaxStrategy());
        return this;
    }

    public HybridMultiDataSetNormalizer minMaxScaleInput(int input, double rangeFrom, double rangeTo) {
        perInputStrategies.put(input, new MinMaxStrategy(rangeFrom, rangeTo));
        return this;
    }

    public HybridMultiDataSetNormalizer standardizeAllOutputs() {
        globalOutputStrategy = new StandardizeStrategy();
        return this;
    }

    public HybridMultiDataSetNormalizer minMaxScaleAllOutputs() {
        globalOutputStrategy = new MinMaxStrategy();
        return this;
    }

    public HybridMultiDataSetNormalizer minMaxScaleAllOutputs(double rangeFrom, double rangeTo) {
        globalOutputStrategy = new MinMaxStrategy(rangeFrom, rangeTo);
        return this;
    }

    public HybridMultiDataSetNormalizer standardizeOutput(int output) {
        perOutputStrategies.put(output, new StandardizeStrategy());
        return this;
    }

    public HybridMultiDataSetNormalizer minMaxScaleOutput(int output) {
        perOutputStrategies.put(output, new MinMaxStrategy());
        return this;
    }

    public HybridMultiDataSetNormalizer minMaxScaleOutput(int output, double rangeFrom, double rangeTo) {
        perOutputStrategies.put(output, new MinMaxStrategy(rangeFrom, rangeTo));
        return this;
    }
    
    public NormalizerStats getInputStats(int input) {
        return inputStats.get(input);
    }

    public NormalizerStats getOutputStats(int output) {
        return outputStats.get(output);
    }

    @Override
    public void fit(@NonNull MultiDataSet dataSet) {
        Map<Integer, NormalizerStats.Builder> inputStatsBuilders = new HashMap<>();
        Map<Integer, NormalizerStats.Builder> outputStatsBuilders = new HashMap<>();

        fitPartial(dataSet, inputStatsBuilders, outputStatsBuilders);

        inputStats = buildAllStats(inputStatsBuilders);
        outputStats = buildAllStats(outputStatsBuilders);
    }

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

    @Override
    public void transform(@NonNull MultiDataSet data) {
        preProcess(data);
    }

    @Override
    public void preProcess(@NonNull MultiDataSet data) {
        preProcess(
            data.getFeatures(),
            data.getFeaturesMaskArrays(),
            globalInputStrategy,
            perInputStrategies,
            inputStats
        );
        preProcess(
            data.getLabels(),
            data.getLabelsMaskArrays(),
            globalOutputStrategy,
            perOutputStrategies,
            outputStats
        );
    }

    private void preProcess(INDArray[] arrays, INDArray[] masks, NormalizerStrategy globalStrategy,
                            Map<Integer, NormalizerStrategy> perArrayStrategy, Map<Integer, NormalizerStats> stats) {

        for (int i = 0; i < arrays.length; i++) {
            NormalizerStrategy strategy = getStrategy(globalStrategy, perArrayStrategy, i);
            if (strategy != null) {
                //noinspection unchecked
                strategy.preProcess(
                    arrays[i],
                    masks == null ? null : masks[i],
                    stats.get(i)
                );
            }
        }
    }

    @Override
    public void revert(@NonNull MultiDataSet data) {
        revertFeatures(data.getFeatures(), data.getFeaturesMaskArrays());
        revertLabels(data.getLabels(), data.getLabelsMaskArrays());
    }

    @Override
    public void revertFeatures(@NonNull INDArray[] features) {
        revertFeatures(features, null);
    }

    @Override
    public void revertFeatures(@NonNull INDArray[] features, INDArray[] maskArrays) {
        for (int i = 0; i < features.length; i++) {
            INDArray mask = (maskArrays == null ? null : maskArrays[i]);
            revertFeatures(features[i], mask, i);
        }
    }

    public void revertFeatures(@NonNull INDArray features, INDArray mask, int input) {
        NormalizerStrategy strategy = getStrategy(globalInputStrategy, perInputStrategies, input);
        if (strategy != null) {
            //noinspection unchecked
            strategy.revert(features, mask, inputStats.get(input));
        }
    }

    @Override
    public void revertLabels(@NonNull INDArray[] labels) {
        revertLabels(labels, null);
    }

    @Override
    public void revertLabels(@NonNull INDArray[] labels, INDArray[] labelsMask) {
        for (int i = 0; i < labels.length; i++) {
            INDArray mask = (labelsMask == null ? null : labelsMask[i]);
            revertLabels(labels[i], mask, i);
        }
    }

    public void revertLabels(@NonNull INDArray labels, INDArray mask, int output) {
        NormalizerStrategy strategy = getStrategy(globalOutputStrategy, perOutputStrategies, output);
        if (strategy != null) {
            //noinspection unchecked
            strategy.revert(labels, mask, outputStats.get(output));
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
}
