package org.datavec.spark.transform.quality.integer;

import lombok.Getter;
import org.datavec.api.transform.metadata.IntegerMetaData;
import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.transform.quality.columns.IntegerQuality;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.quality.QualityAnalysisState;

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

    public IntegerQualityAnalysisState add(Writable writable) throws Exception {
        integerQuality = addFunction.call(integerQuality, writable);
        return this;
    }

    public IntegerQualityAnalysisState merge(IntegerQualityAnalysisState other) throws Exception {
        integerQuality = mergeFunction.call(integerQuality, other.getIntegerQuality());
        return this;
    }

    public ColumnQuality getColumnQuality() {
        return integerQuality;
    }
}
