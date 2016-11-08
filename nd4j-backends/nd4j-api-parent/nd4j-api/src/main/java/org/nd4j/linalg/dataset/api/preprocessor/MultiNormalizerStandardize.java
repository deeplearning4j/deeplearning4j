package org.nd4j.linalg.dataset.api.preprocessor;

import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DistributionStats;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Pre processor for MultiDataSet that normalizes feature values (and optionally label values) to have 0 mean and
 * a standard deviation of 1
 *
 * @author Ede Meijer
 */
public class MultiNormalizerStandardize extends AbstractNormalizerStandardize implements MultiDataSetPreProcessor {
    private List<DistributionStats> featureStats;
    private List<DistributionStats> labelStats;

    /**
     * Fit the model with a MultiDataSet to calculate means and standard deviations with
     *
     * @param dataSet
     */
    public void fit(@NonNull MultiDataSet dataSet) {
        List<DistributionStats.Builder> featureNormBuilders = new ArrayList<>();
        List<DistributionStats.Builder> labelNormBuilders = new ArrayList<>();

        fitPartial(dataSet, featureNormBuilders, labelNormBuilders);

        featureStats = DistributionStats.Builder.buildList(featureNormBuilders);
        if (fitLabels) {
            labelStats = DistributionStats.Builder.buildList(labelNormBuilders);
        }
    }

    /**
     * FFit the model with a MultiDataSetIterator to calculate means and standard deviations with
     *
     * @param iterator
     */
    public void fit(@NonNull MultiDataSetIterator iterator) {
        List<DistributionStats.Builder> featureNormBuilders = new ArrayList<>();
        List<DistributionStats.Builder> labelNormBuilders = new ArrayList<>();

        iterator.reset();
        while (iterator.hasNext()) {
            MultiDataSet next = iterator.next();
            fitPartial(next, featureNormBuilders, labelNormBuilders);
        }

        featureStats = DistributionStats.Builder.buildList(featureNormBuilders);
        if (fitLabels) {
            labelStats = DistributionStats.Builder.buildList(labelNormBuilders);
        }
    }

    private void fitPartial(
        MultiDataSet dataSet,
        List<DistributionStats.Builder> featureStatsBuilders,
        List<DistributionStats.Builder> labelStatsBuilders
    ) {
        int numInputs = dataSet.getFeatures().length;
        int numOutputs = dataSet.getLabels().length;

        ensureStatsBuilders(featureStatsBuilders, numInputs);
        ensureStatsBuilders(labelStatsBuilders, numOutputs);

        for (int i = 0; i < numInputs; i++) {
            featureStatsBuilders.get(i).add(dataSet.getFeatures(i), dataSet.getFeaturesMaskArray(i));
        }

        if (fitLabels) {
            for (int i = 0; i < numOutputs; i++) {
                labelStatsBuilders.get(i).add(dataSet.getLabels(i), dataSet.getLabelsMaskArray(i));
            }
        }
    }

    private void ensureStatsBuilders(List<DistributionStats.Builder> builders, int amount) {
        if (builders.isEmpty()) {
            for (int i = 0; i < amount; i++) {
                builders.add(new DistributionStats.Builder());
            }
        }
    }

    @Override
    public void preProcess(@NonNull MultiDataSet toPreProcess) {
        assertIsFit();
        int numFeatures = toPreProcess.getFeatures().length;
        int numLabels = toPreProcess.getLabels().length;

        for (int i = 0; i < numFeatures; i++) {
            preProcess(toPreProcess.getFeatures(i), featureStats.get(i));
        }
        if (fitLabels) {
            for (int i = 0; i < numLabels; i++) {
                preProcess(toPreProcess.getLabels(i), labelStats.get(i));
            }
        }
    }

    /**
     * Revert the data to what it was before transform
     *
     * @param data the dataset to revert back
     */
    public void revert(@NonNull MultiDataSet data) {
        assertIsFit();

        INDArray[] inputs = data.getFeatures();
        for (int i = 0; i < inputs.length; i++) {
            revert(inputs[i], featureStats.get(i));
        }
        if (fitLabels) {
            INDArray[] outputs = data.getLabels();
            for (int i = 0; i < outputs.length; i++) {
                revert(outputs[i], labelStats.get(i));
            }
        }
    }


    public INDArray getFeatureMean(int input) {
        assertIsFit();
        return featureStats.get(input).getMean();
    }

    public INDArray getLabelMean(int output) {
        assertIsFit();
        return labelStats.get(output).getMean();
    }

    public INDArray getFeatureStd(int input) {
        assertIsFit();
        return featureStats.get(input).getStd();
    }

    public INDArray getLabelStd(int output) {
        assertIsFit();
        return labelStats.get(output).getStd();
    }

    @Override
    protected boolean isFit() {
        return featureStats != null;
    }

    /**
     * Load means and standard deviations from the file system
     *
     * @param featureFiles source files for features, requires 2 files per input, alternating mean and stddev files
     * @param labelFiles   source files for labels, requires 2 files per output, alternating mean and stddev files
     */
    public void load(@NonNull List<File> featureFiles, @NonNull List<File> labelFiles) throws IOException {
        featureStats = load(featureFiles);
        if (fitLabels) {
            labelStats = load(labelFiles);
        }
    }

    private List<DistributionStats> load(List<File> files) throws IOException {
        ArrayList<DistributionStats> stats = new ArrayList<>(files.size() / 2);
        for (int i = 0; i < files.size() / 2; i++) {
            stats.add(DistributionStats.load(files.get(i * 2), files.get(i * 2 + 1)));
        }
        return stats;
    }

    /**
     * Save the current means and standard deviations to the file system
     *
     * @param featureFiles target files for features, requires 2 files per input, alternating mean and stddev files
     * @param labelFiles   target files for labels, requires 2 files per output, alternating mean and stddev files
     */
    public void save(@NonNull List<File> featureFiles, @NonNull List<File> labelFiles) throws IOException {
        saveStats(featureStats, featureFiles);
        if (fitLabels) {
            saveStats(labelStats, labelFiles);
        }
    }

    private void saveStats(List<DistributionStats> stats, List<File> files) throws IOException {
        int requiredFiles = stats.size() * 2;
        if (requiredFiles != files.size()) {
            throw new RuntimeException(String.format(
                "Need twice as many files as inputs / outputs (%d), got %d",
                requiredFiles,
                files.size()
            ));
        }

        for (int i = 0; i < stats.size(); i++) {
            stats.get(i).save(files.get(i * 2), files.get(i * 2 + 1));
        }
    }
}
