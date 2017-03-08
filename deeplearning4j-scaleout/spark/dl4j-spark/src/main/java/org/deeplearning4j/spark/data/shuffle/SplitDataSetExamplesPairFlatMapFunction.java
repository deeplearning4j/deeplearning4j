package org.deeplearning4j.spark.data.shuffle;

import org.apache.spark.api.java.JavaRDD;
import org.datavec.spark.functions.FlatMapFunctionAdapter;
import org.deeplearning4j.spark.util.BasePairFlatMapFunctionAdaptee;
import org.nd4j.linalg.dataset.DataSet;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A PairFlatMapFunction that splits each example in a {@link DataSet} object into its own {@link DataSet}.
 * Also adds a random key (integer value) in the range 0 to maxKeyIndex-1.<br>
 *
 * Used in {@link org.deeplearning4j.spark.util.SparkUtils#shuffleExamples(JavaRDD, int, int)}
 *
 * @author Alex Black
 */
public class SplitDataSetExamplesPairFlatMapFunction extends BasePairFlatMapFunctionAdaptee<DataSet, Integer, DataSet> {

    public SplitDataSetExamplesPairFlatMapFunction(int maxKeyIndex) {
        super(new SplitDataSetExamplesPairFlatMapFunctionAdapter(maxKeyIndex));
    }
}


/**
 * A PairFlatMapFunction that splits each example in a {@link DataSet} object into its own {@link DataSet}.
 * Also adds a random key (integer value) in the range 0 to maxKeyIndex-1.<br>
 *
 * Used in {@link org.deeplearning4j.spark.util.SparkUtils#shuffleExamples(JavaRDD, int, int)}
 *
 * @author Alex Black
 */
class SplitDataSetExamplesPairFlatMapFunctionAdapter
                implements FlatMapFunctionAdapter<DataSet, Tuple2<Integer, DataSet>> {

    private transient Random r;
    private int maxKeyIndex;

    public SplitDataSetExamplesPairFlatMapFunctionAdapter(int maxKeyIndex) {
        this.maxKeyIndex = maxKeyIndex;
    }

    @Override
    public Iterable<Tuple2<Integer, DataSet>> call(DataSet dataSet) throws Exception {
        if (r == null) {
            r = new Random();
        }

        List<DataSet> singleExamples = dataSet.asList();
        List<Tuple2<Integer, DataSet>> out = new ArrayList<>(singleExamples.size());
        for (DataSet ds : singleExamples) {
            out.add(new Tuple2<>(r.nextInt(maxKeyIndex), ds));
        }

        return out;
    }
}
