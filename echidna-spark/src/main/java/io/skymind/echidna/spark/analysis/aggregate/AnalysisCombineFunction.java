package io.skymind.echidna.spark.analysis.aggregate;

import io.skymind.echidna.spark.analysis.AnalysisCounter;
import org.apache.spark.api.java.function.Function2;

import java.util.List;

/**
 * Combine function used for undertaking analysis of a data set via Spark
 *
 * @author Alex Black
 */
public class AnalysisCombineFunction implements Function2<List<AnalysisCounter>,List<AnalysisCounter>,List<AnalysisCounter>> {
    @Override
    public List<AnalysisCounter> call(List<AnalysisCounter> l1, List<AnalysisCounter> l2) throws Exception {
        if(l1 == null) return l2;
        if(l2 == null) return l1;

        int size = l1.size();
        if(size != l2.size()) throw new IllegalStateException("List lengths differ");

        for( int i=0; i<size; i++ ){
            l1.get(i).merge(l2.get(i));
        }
        return l1;
    }
}
