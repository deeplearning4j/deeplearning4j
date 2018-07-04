package org.deeplearning4j.datasets.iterator;

import org.nd4j.linalg.factory.Nd4j;

/**
 * RandomDataSetIterator: Generates random values (or zeros, ones, integers, etc) according to some distribution.<br>
 * Note: This is typically used for testing, debugging and benchmarking purposes.
 *
 * @author Alex Black
 */
public class RandomDataSetIterator extends MultiDataSetWrapperIterator {

    public enum Values {RANDOM_UNIFORM, RANDOM_NORMAL, ONE_HOT, ZEROS, ONES, BINARY, INTEGER_0_10, INTEGER_0_100, INTEGER_0_1000,
        INTEGER_0_10000, INTEGER_0_100000;
        public RandomMultiDataSetIterator.Values toMdsValues(){
            return RandomMultiDataSetIterator.Values.valueOf(this.toString());
        }
    };

    public RandomDataSetIterator(int numMiniBatches, long[] featuresShape, long[] labelsShape, Values featureValues, Values labelValues){
        this(numMiniBatches, featuresShape, labelsShape, featureValues, labelValues, Nd4j.order(), Nd4j.order());
    }

    public RandomDataSetIterator(int numMiniBatches, long[] featuresShape, long[] labelsShape, Values featureValues, Values labelValues,
                                 char featuresOrder, char labelsOrder){
        super(new RandomMultiDataSetIterator.Builder(numMiniBatches)
                .addFeatures(featuresShape, featuresOrder, featureValues.toMdsValues())
                .addLabels(labelsShape, labelsOrder, labelValues.toMdsValues())
        .build());
    }

}
