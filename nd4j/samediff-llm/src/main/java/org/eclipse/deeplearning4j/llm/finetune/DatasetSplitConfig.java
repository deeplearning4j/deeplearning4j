package org.eclipse.deeplearning4j.llm.finetune;

/** Immutable deterministic dataset split settings. */
public final class DatasetSplitConfig {
    private final double trainRatio;
    private final double validationRatio;
    private final double testRatio;
    private final String salt;

    public DatasetSplitConfig(double trainRatio, double validationRatio, double testRatio, String salt) {
        if (trainRatio < 0 || validationRatio < 0 || testRatio < 0
                || Math.abs(trainRatio + validationRatio + testRatio - 1.0) > 1e-9) {
            throw new IllegalArgumentException("Split ratios must be non-negative and sum to 1");
        }
        this.trainRatio = trainRatio;
        this.validationRatio = validationRatio;
        this.testRatio = testRatio;
        this.salt = salt == null ? "" : salt;
    }

    public static DatasetSplitConfig defaultSplit() {
        return new DatasetSplitConfig(0.8, 0.1, 0.1, "");
    }

    public double getTrainRatio() { return trainRatio; }
    public double getValidationRatio() { return validationRatio; }
    public double getTestRatio() { return testRatio; }
    public String getSalt() { return salt; }
}
