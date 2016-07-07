package org.datavec.spark.transform.analysis.string;

import org.apache.spark.api.java.function.Function2;

/**
 * Created by Alex on 5/03/2016.
 */
public class StringAnalysisMergeFunction implements Function2<StringAnalysisCounter,StringAnalysisCounter,StringAnalysisCounter> {
    @Override
    public StringAnalysisCounter call(StringAnalysisCounter v1, StringAnalysisCounter v2) throws Exception {
        return v1.merge(v2);
    }
}
