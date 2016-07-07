package org.datavec.spark.transform.analysis.histogram;

import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.schema.Schema;
import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function2;
import org.datavec.api.writable.Writable;

import java.util.ArrayList;
import java.util.List;

/**
 * An adder function used in the calculation of histograms
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class HistogramAddFunction implements Function2<List<HistogramCounter>,List<Writable>,List<HistogramCounter>> {
    private final int nBins;
    private final Schema schema;
    private final double[][] minsMaxes;

    @Override
    public List<HistogramCounter> call(List<HistogramCounter> histogramCounters, List<Writable> writables) throws Exception {
        if(histogramCounters == null){
            histogramCounters = new ArrayList<>();
            List<ColumnType> columnTypes = schema.getColumnTypes();
            int i=0;
            for(ColumnType ct : columnTypes){
                switch (ct){
                    case String:
                        histogramCounters.add(new StringHistogramCounter((int)minsMaxes[i][0], (int)minsMaxes[i][1], nBins));
                        break;
                    case Integer:
                        histogramCounters.add(new DoubleHistogramCounter(minsMaxes[i][0], minsMaxes[i][1], nBins));
                        break;
                    case Long:
                        histogramCounters.add(new DoubleHistogramCounter(minsMaxes[i][0], minsMaxes[i][1], nBins));
                        break;
                    case Double:
                        histogramCounters.add(new DoubleHistogramCounter(minsMaxes[i][0], minsMaxes[i][1], nBins));
                        break;
                    case Categorical:
                        histogramCounters.add(null);    //TODO
                        break;
                    case Time:
                        histogramCounters.add(new DoubleHistogramCounter(minsMaxes[i][0], minsMaxes[i][1], nBins));
                        break;
                    case Bytes:
                        histogramCounters.add(null);    //TODO
                        break;
                    default:
                        throw new IllegalArgumentException("Unknown column type: " + ct);
                }

                i++;
            }
        }

        int size = histogramCounters.size();
        if(size != writables.size()) throw new IllegalStateException("Writables list and number of counters does not match (" + writables.size() + " vs " + size + ")");
        for( int i=0; i<size; i++ ){
            HistogramCounter hc = histogramCounters.get(i);
            if(hc != null) hc.add(writables.get(i));
        }

        return histogramCounters;
    }
}
