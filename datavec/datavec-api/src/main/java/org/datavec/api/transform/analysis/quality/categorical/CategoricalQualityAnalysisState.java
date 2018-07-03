package org.datavec.api.transform.analysis.quality.categorical;

import lombok.Getter;
import org.datavec.api.transform.analysis.quality.QualityAnalysisState;
import org.datavec.api.transform.metadata.CategoricalMetaData;
import org.datavec.api.transform.quality.columns.CategoricalQuality;
import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.writable.Writable;

/**
 * Created by huitseeker on 3/6/17.
 */
public class CategoricalQualityAnalysisState implements QualityAnalysisState<CategoricalQualityAnalysisState> {

    @Getter
    private CategoricalQuality categoricalQuality;
    private CategoricalQualityAddFunction addFunction;
    private CategoricalQualityMergeFunction mergeFunction;

    public CategoricalQualityAnalysisState(CategoricalMetaData integerMetaData) {
        this.categoricalQuality = new CategoricalQuality();
        this.addFunction = new CategoricalQualityAddFunction(integerMetaData);
        this.mergeFunction = new CategoricalQualityMergeFunction();
    }

    public CategoricalQualityAnalysisState add(Writable writable) {
        categoricalQuality = addFunction.apply(categoricalQuality, writable);
        return this;
    }

    public CategoricalQualityAnalysisState merge(CategoricalQualityAnalysisState other) {
        categoricalQuality = mergeFunction.apply(categoricalQuality, other.getCategoricalQuality());
        return this;
    }


    public ColumnQuality getColumnQuality() {
        return categoricalQuality;
    }
}
