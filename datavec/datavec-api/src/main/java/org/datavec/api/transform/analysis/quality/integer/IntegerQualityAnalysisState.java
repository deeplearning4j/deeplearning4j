package org.datavec.api.transform.analysis.quality.integer;

import lombok.Getter;
import org.datavec.api.transform.analysis.quality.QualityAnalysisState;
import org.datavec.api.transform.metadata.IntegerMetaData;
import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.transform.quality.columns.IntegerQuality;
import org.datavec.api.writable.Writable;

/**
 * Created by huitseeker on 3/6/17.
 */
public class IntegerQualityAnalysisState implements QualityAnalysisState<IntegerQualityAnalysisState> {

    @Getter
    private IntegerQuality integerQuality;
    private IntegerQualityAddFunction addFunction;
    private IntegerQualityMergeFunction mergeFunction;

    public IntegerQualityAnalysisState(IntegerMetaData integerMetaData) {
        this.integerQuality = new IntegerQuality(0, 0, 0, 0, 0);
        this.addFunction = new IntegerQualityAddFunction(integerMetaData);
        this.mergeFunction = new IntegerQualityMergeFunction();
    }

    public IntegerQualityAnalysisState add(Writable writable) {
        integerQuality = addFunction.apply(integerQuality, writable);
        return this;
    }

    public IntegerQualityAnalysisState merge(IntegerQualityAnalysisState other) {
        integerQuality = mergeFunction.apply(integerQuality, other.getIntegerQuality());
        return this;
    }

    public ColumnQuality getColumnQuality() {
        return integerQuality;
    }
}
