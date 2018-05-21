package org.datavec.spark.transform.quality.real;

import lombok.Getter;
import org.datavec.api.transform.metadata.DoubleMetaData;
import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.transform.quality.columns.DoubleQuality;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.quality.QualityAnalysisState;

/**
 * Created by huitseeker on 3/6/17.
 */
public class RealQualityAnalysisState implements QualityAnalysisState<RealQualityAnalysisState> {

    @Getter
    private DoubleQuality realQuality;
    private RealQualityAddFunction addFunction;
    private RealQualityMergeFunction mergeFunction;

    public RealQualityAnalysisState(DoubleMetaData realMetaData) {
        this.realQuality = new DoubleQuality();
        this.addFunction = new RealQualityAddFunction(realMetaData);
        this.mergeFunction = new RealQualityMergeFunction();
    }

    public RealQualityAnalysisState add(Writable writable) throws Exception {
        realQuality = addFunction.call(realQuality, writable);
        return this;
    }

    public RealQualityAnalysisState merge(RealQualityAnalysisState other) throws Exception {
        realQuality = mergeFunction.call(realQuality, other.getRealQuality());
        return this;
    }

    public ColumnQuality getColumnQuality() {
        return realQuality;
    }
}
