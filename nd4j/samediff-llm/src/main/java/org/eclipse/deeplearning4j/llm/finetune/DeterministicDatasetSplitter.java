package org.eclipse.deeplearning4j.llm.finetune;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Stable hash-based splitting, optionally grouping related examples to prevent leakage. */
public final class DeterministicDatasetSplitter {
    @FunctionalInterface
    public interface GroupKey {
        String group(GeneratedTrainingExample example);
    }

    private DeterministicDatasetSplitter() {}

    public static DatasetSplit split(List<GeneratedTrainingExample> examples,
                                     DatasetSplitConfig config, GroupKey groupKey) {
        if (examples == null || config == null) throw new IllegalArgumentException("examples and config are required");
        List<GeneratedTrainingExample> ordered = new ArrayList<>(examples);
        ordered.sort(Comparator.comparing(GeneratedTrainingExample::getId));
        Set<String> ids = new HashSet<>();
        List<GeneratedTrainingExample> train = new ArrayList<>();
        List<GeneratedTrainingExample> validation = new ArrayList<>();
        List<GeneratedTrainingExample> test = new ArrayList<>();
        for (GeneratedTrainingExample example : ordered) {
            example.validate();
            if (!ids.add(example.getId())) throw new IllegalArgumentException("Duplicate example id: " + example.getId());
            String group = groupKey == null ? example.getId() : groupKey.group(example);
            double bucket = bucket(config.getSalt() + "\n" + (group == null ? example.getId() : group));
            if (bucket < config.getTrainRatio()) train.add(example);
            else if (bucket < config.getTrainRatio() + config.getValidationRatio()) validation.add(example);
            else test.add(example);
        }
        return new DatasetSplit(train, validation, test);
    }

    private static double bucket(String value) {
        try {
            byte[] hash = MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8));
            long positive = 0;
            for (int i = 0; i < 8; i++) positive = (positive << 8) | (hash[i] & 0xffL);
            return (positive >>> 1) / (double) Long.MAX_VALUE;
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }
}
