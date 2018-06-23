package org.datavec.local.transforms;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.analysis.AnalysisCounter;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.analysis.DataVecAnalysisUtils;
import org.datavec.api.transform.analysis.columns.ColumnAnalysis;
import org.datavec.api.transform.analysis.histogram.HistogramCounter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.local.transforms.analysis.aggregate.AnalysisAddFunction;
import org.datavec.local.transforms.analysis.histogram.HistogramAddFunction;

import java.util.List;

public class AnalyzeLocal {
    private static final int DEFAULT_MAX_HISTOGRAM_BUCKETS = 20;

    DataAnalysis analyze(Schema schema, RecordReader rr) {
        return analyze(schema, rr, DEFAULT_MAX_HISTOGRAM_BUCKETS);
    }

    DataAnalysis analyze(Schema schema, RecordReader rr, int maxHistogramBuckets){
        AnalysisAddFunction addFn = new AnalysisAddFunction(schema);
        List<AnalysisCounter> counters = null;
        while(rr.hasNext()){
            counters = addFn.apply(counters, rr.next());
        }

        double[][] minsMaxes = new double[counters.size()][2];

        List<ColumnType> columnTypes = schema.getColumnTypes();
        List<ColumnAnalysis> list = DataVecAnalysisUtils.convertCounters(counters, minsMaxes, columnTypes);


        //Do another pass collecting histogram values:
        List<HistogramCounter> histogramCounters = null;
        HistogramAddFunction add = new HistogramAddFunction(maxHistogramBuckets, schema, minsMaxes);
        if(rr.resetSupported()){
            rr.reset();
            while(rr.hasNext()){
                histogramCounters = add.apply(histogramCounters, rr.next());
            }

            DataVecAnalysisUtils.mergeCounters(list, histogramCounters);
        }

        return new DataAnalysis(schema, list);
    }

}
