package org.nd4j.etl4j.spark.transform.analysis.histogram;

import org.apache.spark.api.java.function.Function2;

import java.util.List;

/**
 * A combiner function used in the calculation of histograms
 *
 * @author Alex Black
 */
public class HistogramCombineFunction implements Function2<List<HistogramCounter>,List<HistogramCounter>,List<HistogramCounter>> {
    @Override
    public List<HistogramCounter> call(List<HistogramCounter> l1, List<HistogramCounter> l2) throws Exception {
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
