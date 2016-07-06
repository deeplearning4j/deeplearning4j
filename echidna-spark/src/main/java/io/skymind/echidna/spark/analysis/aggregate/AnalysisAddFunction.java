package io.skymind.echidna.spark.analysis.aggregate;

import io.skymind.echidna.api.ColumnType;
import io.skymind.echidna.api.schema.Schema;
import io.skymind.echidna.spark.analysis.AnalysisCounter;
import io.skymind.echidna.spark.analysis.columns.BytesAnalysisCounter;
import io.skymind.echidna.spark.analysis.columns.CategoricalAnalysisCounter;
import io.skymind.echidna.spark.analysis.columns.IntegerAnalysisCounter;
import io.skymind.echidna.spark.analysis.columns.LongAnalysisCounter;
import io.skymind.echidna.spark.analysis.columns.DoubleAnalysisCounter;
import io.skymind.echidna.spark.analysis.string.StringAnalysisCounter;
import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function2;
import org.canova.api.writable.Writable;

import java.util.ArrayList;
import java.util.List;

/**
 * Add function used for undertaking analysis of a data set via Spark
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class AnalysisAddFunction implements Function2<List<AnalysisCounter>,List<Writable>,List<AnalysisCounter>> {

    private Schema schema;

    @Override
    public List<AnalysisCounter> call(List<AnalysisCounter> analysisCounters, List<Writable> writables) throws Exception {
        if(analysisCounters == null){
            analysisCounters = new ArrayList<>();
            List<ColumnType> columnTypes = schema.getColumnTypes();
            for(ColumnType ct : columnTypes){
                switch (ct){
                    case String:
                        analysisCounters.add(new StringAnalysisCounter());
                        break;
                    case Integer:
                        analysisCounters.add(new IntegerAnalysisCounter());
                        break;
                    case Long:
                        analysisCounters.add(new LongAnalysisCounter());
                        break;
                    case Double:
                        analysisCounters.add(new DoubleAnalysisCounter());
                        break;
                    case Categorical:
                        analysisCounters.add(new CategoricalAnalysisCounter());
                        break;
                    case Time:
                        analysisCounters.add(new LongAnalysisCounter());
                        break;
                    case Bytes:
                        analysisCounters.add(new BytesAnalysisCounter());
                        break;
                    default:
                        throw new IllegalArgumentException("Unknown column type: " + ct);
                }
            }
        }

        int size = analysisCounters.size();
        if(size != writables.size()) throw new IllegalStateException("Writables list and number of counters does not match (" + writables.size() + " vs " + size + ")");
        for( int i=0; i<size; i++ ){
            analysisCounters.get(i).add(writables.get(i));
        }

        return analysisCounters;
    }
}
