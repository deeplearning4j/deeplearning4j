package org.datavec.api.transform.analysis.quality.string;

import lombok.Getter;
import org.datavec.api.transform.analysis.quality.QualityAnalysisState;
import org.datavec.api.transform.metadata.StringMetaData;
import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.transform.quality.columns.StringQuality;
import org.datavec.api.writable.Writable;

/**
 * Created by huitseeker on 3/6/17.
 */
public class StringQualityAnalysisState implements QualityAnalysisState<StringQualityAnalysisState> {

    @Getter
    private StringQuality stringQuality;
    private StringQualityAddFunction addFunction;
    private StringQualityMergeFunction mergeFunction;

    public StringQualityAnalysisState(StringMetaData stringMetaData) {
        this.stringQuality = new StringQuality();
        this.addFunction = new StringQualityAddFunction(stringMetaData);
        this.mergeFunction = new StringQualityMergeFunction();
    }

    public StringQualityAnalysisState add(Writable writable) {
        stringQuality = addFunction.apply(stringQuality, writable);
        return this;
    }

    public StringQualityAnalysisState merge(StringQualityAnalysisState other) {
        stringQuality = mergeFunction.apply(stringQuality, other.getStringQuality());
        return this;
    }

    public ColumnQuality getColumnQuality() {
        return stringQuality;
    }
}
