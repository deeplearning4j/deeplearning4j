package org.deeplearning4j.spark.data;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Take an existing DataSet object, and split it into multiple DataSet objects with one example in each
 *
 * Usage:
 * <pre>
 * {@code
 *      RDD<DataSet> myBatchedExampleDataSets = ...;
 *      RDD<DataSet> singleExamlpeDataSets = myBatchedExampleDataSets.mapPartitions(new SplitDataSets(batchSize));
 * }
 * </pre>
 *
 * @author Alex Black
 */
public class SplitDataSetsFunction implements FlatMapFunction<Iterator<DataSet>,DataSet> {
    @Override
    public Iterable<DataSet> call(Iterator<DataSet> dataSetIterator) throws Exception {
        List<DataSet> out = new ArrayList<>();
        while(dataSetIterator.hasNext()){
            out.addAll(dataSetIterator.next().asList());
        }
        return out;
    }
}
