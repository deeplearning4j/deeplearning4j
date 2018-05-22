package org.datavec.spark.transform.quality.longq;

import lombok.Getter;
import org.datavec.api.transform.metadata.LongMetaData;
import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.transform.quality.columns.LongQuality;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.quality.QualityAnalysisState;

/**
 * Created by huitseeker on 3/6/17.
 */
public class LongQualityAnalysisState implements QualityAnalysisState<LongQualityAnalysisState> {

    @Getter
    private LongQuality longQuality;
    private LongQualityAddFunction addFunction;
    private LongQualityMergeFunction mergeFunction;

    public LongQualityAnalysisState(LongMetaData longMetaData) {
        this.longQuality = new LongQuality();
        this.addFunction = new LongQualityAddFunction(longMetaData);
        this.mergeFunction = new LongQualityMergeFunction();
    }

    public LongQualityAnalysisState add(Writable writable) throws Exception {
        longQuality = addFunction.call(longQuality, writable);
        return this;
    }

    public LongQualityAnalysisState merge(LongQualityAnalysisState other) throws Exception {
        longQuality = mergeFunction.call(longQuality, other.getLongQuality());
        return this;
    }

    public ColumnQuality getColumnQuality() {
        return longQuality;
    }
}
