package org.deeplearning4j.spark.canova;

import org.apache.spark.api.java.function.Function;
import org.canova.api.io.WritableConverter;
import org.canova.api.writable.Writable;
import org.canova.common.data.NDArrayWritable;
import org.deeplearning4j.datasets.canova.SequenceRecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.FeatureUtil;

import java.io.Serializable;
import java.util.Collection;
import java.util.Iterator;

/**Map {@code Collection<Collection<Writable>>} objects (out of a canova-spark sequence record reader function) to
 *  DataSet objects for Spark training.
 * Analogous to {@link SequenceRecordReaderDataSetIterator}, but in the context of Spark.
 * Supports loading data from a single source only (hence no masknig arrays, many-to-one etc here)
 * see {@link CanovaSequencePairDataSetFunction} for the separate collections for input and labels version
 * @author Alex Black
 */
public class CanovaSequenceDataSetFunction implements Function<Collection<Collection<Writable>>,DataSet>, Serializable {

    private final boolean regression;
    private final int labelIndex;
    private final int numPossibleLabels;
    private final DataSetPreProcessor preProcessor;
    private final WritableConverter converter;

    /**
     * @param labelIndex Index of the label column
     * @param numPossibleLabels Number of classes for classification  (not used if regression = true)
     * @param regression False for classification, true for regression
     */
    public CanovaSequenceDataSetFunction(int labelIndex, int numPossibleLabels, boolean regression){
        this(labelIndex, numPossibleLabels, regression, null, null);
    }

    /**
     * @param labelIndex Index of the label column
     * @param numPossibleLabels Number of classes for classification  (not used if regression = true)
     * @param regression False for classification, true for regression
     * @param preProcessor DataSetPreprocessor (may be null)
     * @param converter WritableConverter (may be null)
     */
    public CanovaSequenceDataSetFunction(int labelIndex, int numPossibleLabels, boolean regression,
                                         DataSetPreProcessor preProcessor, WritableConverter converter){
        this.labelIndex = labelIndex;
        this.numPossibleLabels = numPossibleLabels;
        this.regression = regression;
        this.preProcessor = preProcessor;
        this.converter = converter;
    }


    @Override
    public DataSet call(Collection<Collection<Writable>> input) throws Exception {
        Iterator<Collection<Writable>> iter = input.iterator();

        INDArray features = null;
        INDArray labels = Nd4j.zeros(1, (regression ? 1 : numPossibleLabels), input.size());

        int[] fIdx = new int[3];
        int[] lIdx = new int[3];

        int i=0;
        while(iter.hasNext()){
            Collection<Writable> step = iter.next();
            if (i == 0) {
                features = Nd4j.zeros(1, step.size()-1, input.size());
            }

            Iterator<Writable> timeStepIter = step.iterator();
            int countIn = 0;
            int countFeatures = 0;
            while (timeStepIter.hasNext()) {
                Writable current = timeStepIter.next();
                if(converter != null) current = converter.convert(current);
                if(countIn++ == labelIndex){
                    //label
                    if(regression){
                        lIdx[2] = i;
                        labels.putScalar(lIdx,current.toDouble());
                    } else {
                        INDArray line = FeatureUtil.toOutcomeVector(current.toInt(), numPossibleLabels);
                        labels.tensorAlongDimension(i,1).assign(line);  //1d from [1,nOut,timeSeriesLength] -> tensor i along dimension 1 is at time i
                    }
                } else {
                    //feature
                    fIdx[1] = countFeatures++;
                    fIdx[2] = i;
                    try {
                        features.putScalar(fIdx, current.toDouble());
                    } catch (UnsupportedOperationException e) {
                        // This isn't a scalar, so check if we got an array already
                        if (current instanceof NDArrayWritable) {
                            features.get(NDArrayIndex.point(fIdx[0]), NDArrayIndex.all(), NDArrayIndex.point(fIdx[2]))
                                    .putRow(0, ((NDArrayWritable)current).get());
                        } else {
                            throw e;
                        }
                    }
                }
            }
            i++;
        }

        DataSet ds = new DataSet(features,labels);
        if(preProcessor != null) preProcessor.preProcess(ds);
        return ds;
    }
}
