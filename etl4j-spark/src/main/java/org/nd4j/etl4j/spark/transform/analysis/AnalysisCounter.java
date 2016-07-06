package org.nd4j.etl4j.spark.transform.analysis;

import org.canova.api.writable.Writable;

import java.io.Serializable;

/**
 * Created by Alex on 23/06/2016.
 */
public interface AnalysisCounter<T extends AnalysisCounter> extends Serializable {

    T add(Writable writable);

    T merge(T other);

}
