package org.nd4j.autodiff.loss;

/**
 * The LossReduce enum specifies how (or if) the values of a loss function should be reduced to a single value.
 * See the javadoc comments on the individual enumeration constants for details.
 *
 * @author Alex Black
 */
public enum LossReduce {
    /**
     * No reduction. In most cases, output is the same shape as the predictions/labels.<br>
     * Weights (if any) are applied<br>
     * Example Input: 2d input array with mean squared error loss.<br>
     * Example Output: squaredDifference(predictions,labels), with same shape as input/labels<br>
     */
    NONE,

    /**
     * Weigted sum across all loss values, returning a scalar.<br>
     */
    SUM,

    /**
     * Weighted mean: sum(weights * perOutputLoss) / sum(weights) - gives a single scalar output<br>
     * Example: 2d input, mean squared error<br>
     * Output: squared_error_per_ex = weights * squaredDifference(predictions,labels)<br>
     *         output = sum(squared_error_per_ex) / sum(weights)<br>
     * <br>
     * NOTE: if weights array is not provided, then weights default to 1.0 for all entries - and hence
     * MEAN_BY_WEIGHT is equivalent to MEAN_BY_NONZERO_WEIGHT_COUNT
     */
    MEAN_BY_WEIGHT,

    /**
     * Weighted mean: sum(weights * perOutputLoss) / count(weights != 0)<br>
     * Example: 2d input, mean squared error loss.<br>
     * Output: squared_error_per_ex = weights * squaredDifference(predictions,labels)<br>
     *         output = sum(squared_error_per_ex) /  count(weights != 0)<br>
     *
     * NOTE: if weights array is not provided, then weights default to scalar 1.0 and hence MEAN_BY_NONZERO_WEIGHT_COUNT
     * is equivalent to MEAN_BY_WEIGHT
     */
    MEAN_BY_NONZERO_WEIGHT_COUNT
}
