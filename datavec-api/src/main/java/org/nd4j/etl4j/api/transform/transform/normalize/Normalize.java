package org.nd4j.etl4j.api.transform.transform.normalize;

/**Built-in normalization methods.
 *
 * Normalization methods include:<br>
 * <b>MinMax</b>: (x-min)/(max-min) -> maps values to range [0,1]<br>
 * <b>MinMax2</b>: 2 * (x-min)/(max-min) + 1 -> maps values to range [-1,1]<br>
 * <b>Standardize</b>: Normalize such that output has distribution N(0,1)<br>
 * <b>SubtractMean</b>: Normalize by only subtracting the mean value<br>
 * <b>Log2Mean</b>: Normalization of the form log2((x-min)/(mean-min) + 1)<br>
 * <b>Log2MeanExcludingMin</b>: As per Log2Mean, but the 'mean' is calculated excluding the minimum value.<br>
 *
 *
 * @author Alex Black
 */
public enum Normalize {

    MinMax,
    MinMax2,
    Standardize,
    SubtractMean,
    Log2Mean,
    Log2MeanExcludingMin

}
