package org.datavec.spark.transform.quality.categorical;

import lombok.Getter;
import org.datavec.api.transform.metadata.CategoricalMetaData;
import org.datavec.api.transform.quality.columns.CategoricalQuality;
import org.datavec.api.transform.quality.columns.ColumnQuality;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.quality.QualityAnalysisState;

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

    public CategoricalQualityAnalysisState add(Writable writable) throws Exception {
        categoricalQuality = addFunction.call(categoricalQuality, writable);
        return this;
    }

    public CategoricalQualityAnalysisState merge(CategoricalQualityAnalysisState other) throws Exception {
        categoricalQuality = mergeFunction.call(categoricalQuality, other.getCategoricalQuality());
        return this;
    }


    public ColumnQuality getColumnQuality() {
        return categoricalQuality;
    }
}
