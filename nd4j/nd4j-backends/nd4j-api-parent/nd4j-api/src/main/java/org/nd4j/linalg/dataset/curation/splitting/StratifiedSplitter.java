package org.nd4j.linalg.dataset.curation.splitting;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;

/**
 * Train/validation/test splitting with stratification support.
 * Supports label-stratified, length-stratified, and random splitting.
 *
 * @param <T> the type of data items
 */
public class StratifiedSplitter<T> implements Serializable {
    private static final long serialVersionUID = 1L;

    private final long seed;

    public StratifiedSplitter(long seed) {
        this.seed = seed;
    }

    public StratifiedSplitter() {
        this(42L);
    }

    /**
     * Random split without stratification.
     */
    public SplitResult<T> split(List<T> data, double trainRatio, double valRatio) {
        double testRatio = 1.0 - trainRatio - valRatio;
        if (testRatio < -0.001) {
            throw new IllegalArgumentException("trainRatio + valRatio must be <= 1.0");
        }

        List<T> shuffled = new ArrayList<>(data);
        Collections.shuffle(shuffled, new Random(seed));

        int trainEnd = (int) (shuffled.size() * trainRatio);
        int valEnd = trainEnd + (int) (shuffled.size() * valRatio);

        List<T> train = new ArrayList<>(shuffled.subList(0, trainEnd));
        List<T> val = new ArrayList<>(shuffled.subList(trainEnd, valEnd));
        List<T> test = new ArrayList<>(shuffled.subList(valEnd, shuffled.size()));

        return new SplitResult<>(train, val, test);
    }

    /**
     * Stratified split: maintain class proportions across splits.
     *
     * @param data the data to split
     * @param trainRatio proportion for training
     * @param valRatio proportion for validation
     * @param labelFn function to extract the stratification key
     */
    public <K> SplitResult<T> splitStratified(List<T> data, double trainRatio, double valRatio, Function<T, K> labelFn) {
        // Group by label
        Map<K, List<T>> groups = new HashMap<>();
        for (T item : data) {
            K label = labelFn.apply(item);
            groups.computeIfAbsent(label, k -> new ArrayList<>()).add(item);
        }

        List<T> train = new ArrayList<>();
        List<T> val = new ArrayList<>();
        List<T> test = new ArrayList<>();

        Random rng = new Random(seed);

        for (Map.Entry<K, List<T>> entry : groups.entrySet()) {
            List<T> group = new ArrayList<>(entry.getValue());
            Collections.shuffle(group, rng);

            int trainEnd = (int) (group.size() * trainRatio);
            int valEnd = trainEnd + (int) (group.size() * valRatio);

            train.addAll(group.subList(0, trainEnd));
            val.addAll(group.subList(trainEnd, valEnd));
            test.addAll(group.subList(valEnd, group.size()));
        }

        // Shuffle the combined splits to mix labels
        Collections.shuffle(train, new Random(seed));
        Collections.shuffle(val, new Random(seed + 1));
        Collections.shuffle(test, new Random(seed + 2));

        return new SplitResult<>(train, val, test);
    }
}
