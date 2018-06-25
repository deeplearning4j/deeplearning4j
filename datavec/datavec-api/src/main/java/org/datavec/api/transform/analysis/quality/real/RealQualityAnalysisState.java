package org.datavec.api.transform.analysis.quality.real;

import lombok.Getter;
import org.datavec.api.transform.analysis.quality.QualityAnalysisState;
import org.datavec.api.transform.metadata.DoubleMetaData;
import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.transform.quality.columns.DoubleQuality;
import org.datavec.api.writable.Writable;

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

    public RealQualityAnalysisState add(Writable writable) {
        realQuality = addFunction.apply(realQuality, writable);
        return this;
    }

    public RealQualityAnalysisState merge(RealQualityAnalysisState other) {
        realQuality = mergeFunction.apply(realQuality, other.getRealQuality());
        return this;
    }

    public ColumnQuality getColumnQuality() {
        return realQuality;
    }
}
