package org.nd4j.etl4j.spark.transform.analysis.columns;

import org.nd4j.etl4j.spark.transform.analysis.AnalysisCounter;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.etl4j.api.writable.Writable;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * A counter function for doing analysis on Categorical columns, on Spark
 *
 * @author Alex Black
 */
@AllArgsConstructor @Data
public class CategoricalAnalysisCounter implements AnalysisCounter<CategoricalAnalysisCounter> {

    private Map<String,Long> counts = new HashMap<>();
    private long countTotal = 0;



    public CategoricalAnalysisCounter(){

    }


    @Override
    public CategoricalAnalysisCounter add(Writable writable) {
        String value = writable.toString();

        long newCount = 0;
        if(counts.containsKey(value)){
            newCount = counts.get(value);
        }
        newCount++;
        counts.put(value, newCount);

        countTotal++;

        return this;
    }

    public CategoricalAnalysisCounter merge(CategoricalAnalysisCounter other){
        Set<String> combinedKeySet = new HashSet<>(counts.keySet());
        combinedKeySet.addAll(other.counts.keySet());

        for(String s : combinedKeySet){
            long count = 0;
            if(counts.containsKey(s)) count += counts.get(s);
            if(other.counts.containsKey(s)) count += other.counts.get(s);
            counts.put(s,count);
        }

        countTotal += other.countTotal;

        return this;
    }

}
