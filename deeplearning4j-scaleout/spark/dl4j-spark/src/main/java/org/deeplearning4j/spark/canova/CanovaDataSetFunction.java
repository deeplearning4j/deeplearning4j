package org.deeplearning4j.spark.canova;

import org.apache.spark.api.java.function.Function;
import org.canova.api.io.WritableConverter;
import org.canova.api.io.converters.WritableConverterException;
import org.canova.api.writable.Writable;
import org.canova.common.data.NDArrayWritable;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**Map {@code Collection<Writable>} objects (out of a canova-spark record reader function) to DataSet objects for Spark training.
 * Analogous to {@link RecordReaderDataSetIterator}, but in the context of Spark.
 * @author Alex Black
 */
public class CanovaDataSetFunction implements Function<Collection<Writable>,DataSet>, Serializable {

    private final int labelIndex;
    private final int numPossibleLabels;
    private final boolean regression;
    private final DataSetPreProcessor preProcessor;
    private final WritableConverter converter;
    protected int batchSize = -1;

    public CanovaDataSetFunction(int labelIndex, int numPossibleLabels, boolean regression){
        this(labelIndex, numPossibleLabels, regression, null, null);
    }

    /**
     * @param labelIndex Index of the label column
     * @param numPossibleLabels Number of classes for classification  (not used if regression = true)
     * @param regression False for classification, true for regression
     * @param preProcessor DataSetPreprocessor (may be null)
     * @param converter WritableConverter (may be null)
     */
    public CanovaDataSetFunction(int labelIndex, int numPossibleLabels, boolean regression,
                                 DataSetPreProcessor preProcessor, WritableConverter converter){
        this.labelIndex = labelIndex;
        this.numPossibleLabels = numPossibleLabels;
        this.regression = regression;
        this.preProcessor = preProcessor;
        this.converter = converter;
    }

    @Override
    public DataSet call(Collection<Writable> writables) throws Exception {
        List<Writable> list;
        if(writables instanceof List) list = (List<Writable>)writables;
        else list = new ArrayList<>(writables);

        //allow people to specify label index as -1 and infer the last possible label
        int labelIndex = this.labelIndex;
        if (numPossibleLabels >= 1 && labelIndex < 0) {
            labelIndex = list.size() - 1;
        }

        INDArray label = null;
        INDArray featureVector = null;
        int featureCount = 0;
        for (int j = 0; j < list.size(); j++) {
            Writable current = list.get(j);
            if(converter != null) current = converter.convert(current);
            if (labelIndex >= 0 && j == labelIndex) {
                //Current value is the label
                if (converter != null) {
                    try {
                        current = converter.convert(current);
                    } catch (WritableConverterException e) {
                        e.printStackTrace();
                    }
                }
                if (numPossibleLabels < 1)
                    throw new IllegalStateException("Number of possible labels invalid, must be >= 1");

                if (regression) {
                    label = Nd4j.scalar(current.toDouble());
                } else {
                    //Convert to one-hot vector for
                    int curr = current.toInt();
                    if (curr >= numPossibleLabels)
                        throw new IllegalStateException("Invalid input: class label is " + curr
                            + " with numPossibleLables = " + numPossibleLabels + " (class label must be 0 <= labelIdx < numPossibleLabels)");
                    label = FeatureUtil.toOutcomeVector(curr, numPossibleLabels);
                }
            } else {
                //Current value is not the label
                try {
                    double value = current.toDouble();
                    if (featureVector == null) {
                        featureVector = Nd4j.create(labelIndex >= 0 ? list.size() - 1 : list.size());
                    }
                    featureVector.putScalar(featureCount++, value);
                } catch (UnsupportedOperationException e) {
                    // This isn't a scalar, so check if we got an array already
                    if (current instanceof NDArrayWritable) {
                        assert featureVector == null;
                        featureVector = ((NDArrayWritable)current).get();
                    } else {
                        throw e;
                    }
                }
            }
        }

        DataSet ds = new DataSet(featureVector, (labelIndex >= 0 ? label : featureVector) );
        if(preProcessor != null) preProcessor.preProcess(ds);
        return ds;
    }
}
