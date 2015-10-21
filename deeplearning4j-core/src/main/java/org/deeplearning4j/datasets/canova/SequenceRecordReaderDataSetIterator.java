package org.deeplearning4j.datasets.canova;

import org.canova.api.records.reader.SequenceRecordReader;
import org.canova.api.writable.Writable;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.util.*;

/**
 * Sequence record reader data set iterator
 * Given a record reader (and optionally another record reader for the labels)
 * generate time series (sequence) data sets
 */
public class SequenceRecordReaderDataSetIterator implements DataSetIterator {
    private SequenceRecordReader recordReader;
    private SequenceRecordReader labelsReader;
    private int miniBatchSize = 10;
    private final boolean regression;
    private int labelIndex = -1;
    private final int numPossibleLabels;
    private int cursor = 0;
    private int inputColumns = -1;
    private int totalOutcomes = -1;
    private boolean useStored = false;
    private DataSet stored = null;

    /**
     * Constructor where features and labels come from different RecordReaders (for example, different files)
     */
    public SequenceRecordReaderDataSetIterator(SequenceRecordReader featuresReader, SequenceRecordReader labels,
                                               int miniBatchSize, int numPossibleLabels, boolean regression) {
        this.recordReader = featuresReader;
        this.labelsReader = labels;
        this.miniBatchSize = miniBatchSize;
        this.numPossibleLabels = numPossibleLabels;
        this.regression = regression;
    }

    @Override
    public boolean hasNext() {
        return recordReader.hasNext();
    }

    @Override
    public DataSet next() {
        return next(miniBatchSize);
    }


    @Override
    public DataSet next(int num) {
        if (useStored) {
            useStored = false;
            DataSet temp = stored;
            stored = null;
            return temp;
        }
        if (!hasNext()) throw new NoSuchElementException();

        List<INDArray> featureList = new ArrayList<>(num);
        List<INDArray> labelList = new ArrayList<>(num);
        for (int i = 0; i < num && hasNext(); i++) {

            Collection<Collection<Writable>> featureSequence = recordReader.sequenceRecord();
            Collection<Collection<Writable>> labelSequence = labelsReader.sequenceRecord();

            INDArray features = getFeatures(featureSequence);
            INDArray labels = getLabels(labelSequence); //2d time series, with shape [timeSeriesLength,vectorSize]

            featureList.add(features);
            labelList.add(labels);
        }

        //Convert 2d sequences/time series to 3d minibatch data
        int[] featureShape = new int[3];
        featureShape[0] = featureList.size();   //mini batch size
        featureShape[1] = featureList.get(0).size(1);   //example vector size
        featureShape[2] = featureList.get(0).size(0);   //time series/sequence length

        int[] labelShape = new int[3];
        labelShape[0] = labelList.size();
        labelShape[1] = labelList.get(0).size(1);   //label vector size
        labelShape[2] = labelList.get(0).size(0);   //time series/sequence length

        INDArray featuresOut = Nd4j.create(featureShape);
        INDArray labelsOut = Nd4j.create(labelShape);
        for (int i = 0; i < featureList.size(); i++) {
            featuresOut.tensorAlongDimension(i, 1, 2).assign(featureList.get(i));
            labelsOut.tensorAlongDimension(i, 1, 2).assign(labelList.get(i));
        }

        cursor += featureList.size();
        if (inputColumns == -1) inputColumns = featuresOut.size(1);
        if (totalOutcomes == -1) totalOutcomes = labelsOut.size(1);
        return new DataSet(featuresOut, labelsOut);
    }

    @Override
    public int totalExamples() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public int inputColumns() {
        if (inputColumns != -1) return inputColumns;
        preLoad();
        return inputColumns;
    }

    @Override
    public int totalOutcomes() {
        if (totalOutcomes != -1) return totalOutcomes;
        preLoad();
        return totalOutcomes;
    }

    private void preLoad() {
        stored = next();
        useStored = true;
        inputColumns = stored.getFeatureMatrix().size(1);
        totalOutcomes = stored.getLabels().size(1);
    }

    @Override
    public void reset() {

    }

    @Override
    public int batch() {
        return miniBatchSize;
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {

    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Remove not supported for this iterator");
    }

    private INDArray getFeatures(Collection<Collection<Writable>> features) {

        //Size of the record?
        int[] shape = new int[2]; //[timeSeriesLength,vectorSize]
        shape[0] = features.size();

        Iterator<Collection<Writable>> iter = features.iterator();

        int i = 0;
        INDArray out = null;
        while (iter.hasNext()) {
            Collection<Writable> step = iter.next();
            if (i == 0) {
                shape[1] = step.size();
                out = Nd4j.create(shape);
            }

            Iterator<Writable> timeStepIter = step.iterator();
            int f = 0;
            while (timeStepIter.hasNext()) {
                Writable current = timeStepIter.next();
                double value = Double.valueOf(current.toString());
                out.put(i, f++, value);
            }
            i++;
        }
        return out;
    }

    private INDArray getLabels(Collection<Collection<Writable>> labels) {
        //Size of the record?
        int[] shape = new int[2];   //[timeSeriesLength,vectorSize]
        shape[0] = labels.size();   //time series/sequence length

        Iterator<Collection<Writable>> iter = labels.iterator();

        int i = 0;
        INDArray out = null;
        while (iter.hasNext()) {
            Collection<Writable> step = iter.next();
            if (i == 0) {
                if (regression) {
                    shape[1] = step.size();
                } else {
                    shape[1] = numPossibleLabels;
                }
                out = Nd4j.create(shape);
            }

            Iterator<Writable> timeStepIter = step.iterator();
            int f = 0;
            if (regression) {
                //Load all values
                while (timeStepIter.hasNext()) {
                    Writable current = timeStepIter.next();
                    double value = Double.valueOf(current.toString());
                    out.put(f++, i, value);
                }
            } else {
                //Expect a single value (index) -> convert to one-hot vector
                Writable value = timeStepIter.next();
                int idx = Double.valueOf(value.toString()).intValue();
                INDArray line = FeatureUtil.toOutcomeVector(idx, numPossibleLabels);
                out.getRow(i).assign(line);
            }

            i++;
        }
        return out;
    }
}
