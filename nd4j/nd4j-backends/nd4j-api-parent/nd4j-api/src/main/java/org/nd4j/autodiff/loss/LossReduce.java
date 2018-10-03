package org.nd4j.autodiff.loss;

public enum LossReduce {
    /**
     * No reduction. Output is the same shape as the predictions/labels.
     * Weights (if any) are applied. Dimension args are ignored.
     * Example: 2d input, MSE.
     * Output: sqDiff(predictions,labels) -> shape same as input/labels
     */
    NONE,
    /**
     * Sum across the remaining dimensions, returning a scalar
     * Example: 2d input, MSE loss along dimension 1.
     * Output: mse_per_ex = mean(weights * sqDiff(predictions,labels),1)
     *         output = sum(mse_per_ex)
     */
    SUM,
    /**
     * Weighted mean: sum(weights * loss) / sum(weights)
     * Example: 2d input, MSE loss along dimension 1.
     * Output: mse_per_ex = mean(weights * sqDiff(predictions,labels),1)
     *         output = sum(mse_per_ex) / sum(weights)
     *
     * NOTE: if weights array is not provided, then weights default to (effectively) 1.0 for all entries - and hence
     * MEAN_BY_WEIGHT is equivalent to SUM (as sum(1.0) = 1.0)
     */
    MEAN_BY_WEIGHT,

    /**
     * Weighted mean: sum(weights * loss) / count(weights != 0)
     * Example: 2d input, MSE loss along dimension 1.
     * Output: mse_per_ex = mean(weights * sqDiff(predictions,labels),1)
     *         output = sum(mse_per_ex) / count(weights != 0)
     *
     * NOTE: if weights array is not provided, then weights default to scalar 1.0 and hence MEAN_BY_COUNT
     * is equivalent to MEAN_BY_WEIGHT
     */
    MEAN_BY_NONZERO_WEIGHT_COUNT
}
