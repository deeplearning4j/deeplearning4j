package org.deeplearning4j.spark.canova;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.nd4j.linalg.dataset.DataSet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * RDD mini batch partitioning
 * @author Adam Gibson
 */
public class RDDMiniBatches  implements Serializable {
    private int miniBatches = 10;
    private JavaRDD<DataSet> toSplitJava;

    public RDDMiniBatches(int miniBatches, JavaRDD<DataSet> toSplit) {
        this.miniBatches = miniBatches;
        this.toSplitJava = toSplit;
    }

    public JavaRDD<DataSet> miniBatchesJava() {
        final int batchSize = miniBatches;

        return toSplitJava.mapPartitions(new MiniBatchFunction(batchSize));
    }


    public static class MiniBatchFunction implements FlatMapFunction<Iterator<DataSet>, DataSet> {
        private int batchSize = 10;

        public MiniBatchFunction(int batchSize) {
            this.batchSize = batchSize;
        }

        @Override
        public Iterable<DataSet> call(Iterator<DataSet> dataSetIterator) throws Exception {
            List<DataSet> ret = new ArrayList<>();
            List<DataSet> temp = new ArrayList<>();
            while (dataSetIterator.hasNext()) {
                temp.add(dataSetIterator.next());
                if (temp.size() == batchSize) {
                    ret.add(DataSet.merge(temp));
                    temp.clear();
                }
            }

            if(!temp.isEmpty())
                ret.add(DataSet.merge(temp));

            return ret;
        }

    }


}
