package org.datavec.spark.transform.analysis.columns;

import org.datavec.spark.transform.analysis.AnalysisCounter;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.datavec.api.writable.Writable;

/**
 * A counter function for doing analysis on BytesWritable columns, on Spark
 *
 * @author Alex Black
 */
@AllArgsConstructor @Data
public class BytesAnalysisCounter implements AnalysisCounter<BytesAnalysisCounter> {
    private long countTotal = 0;



    public BytesAnalysisCounter(){

    }


    @Override
    public BytesAnalysisCounter add(Writable writable) {
        countTotal++;

        return this;
    }

    public BytesAnalysisCounter merge(BytesAnalysisCounter other){

        return new BytesAnalysisCounter(countTotal + other.countTotal);
    }

}
