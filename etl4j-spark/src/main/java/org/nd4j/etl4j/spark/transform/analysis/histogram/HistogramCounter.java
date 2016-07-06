package org.nd4j.etl4j.spark.transform.analysis.histogram;

import org.nd4j.etl4j.api.writable.Writable;

import java.io.Serializable;

/**
 * HistogramCounter: used to calculate histogram values for one column
 *
 * @author Alex Black
 */
public interface HistogramCounter extends Serializable {

    HistogramCounter add(Writable w);

    HistogramCounter merge(HistogramCounter other);

    double[] getBins();

    long[] getCounts();

}
