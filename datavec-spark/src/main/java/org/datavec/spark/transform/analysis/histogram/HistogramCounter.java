package org.datavec.spark.transform.analysis.histogram;

import org.datavec.api.writable.Writable;

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
