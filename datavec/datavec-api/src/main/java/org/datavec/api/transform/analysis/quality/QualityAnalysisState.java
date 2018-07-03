package org.datavec.api.transform.analysis.quality;

import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.writable.Writable;

import java.io.Serializable;

/**
 * Created by huitseeker on 3/6/17.
 */
public interface QualityAnalysisState<T extends QualityAnalysisState> extends Serializable {

    T add(Writable writable);

    T merge(T other);

    ColumnQuality getColumnQuality();
}
