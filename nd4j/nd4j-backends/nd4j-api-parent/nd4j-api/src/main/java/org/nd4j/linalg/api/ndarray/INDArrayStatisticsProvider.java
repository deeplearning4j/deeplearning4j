package org.nd4j.linalg.api.ndarray;

/**
 * A provider interface for backend specific computation of ndarray statistics.
 *
 * @author Adam Gibson
 */
public interface INDArrayStatisticsProvider {

    /**
     * Returns {@link INDArrayStatistics} about a given {@link INDArray}
     * @param arr
     * @return
     */
    INDArrayStatistics inspectArray(INDArray arr);

}
