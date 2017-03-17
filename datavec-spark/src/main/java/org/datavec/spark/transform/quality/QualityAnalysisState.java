package org.datavec.spark.transform.quality;

import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.writable.Writable;

import java.io.Serializable;

/**
 * Created by huitseeker on 3/6/17.
 */
public interface QualityAnalysisState<T extends QualityAnalysisState> extends Serializable {

    T add(Writable writable) throws Exception;

    T merge(T other) throws Exception;

    ColumnQuality getColumnQuality();
}
