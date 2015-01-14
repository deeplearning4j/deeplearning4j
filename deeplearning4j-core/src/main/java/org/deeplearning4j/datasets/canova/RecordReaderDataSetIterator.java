package org.deeplearning4j.datasets.canova;

import org.canova.api.io.WritableConverter;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.writable.Writable;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetPreProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;


/**
 * Record reader dataset iterator
 *
 * @author Adam Gibson
 */
public class RecordReaderDataSetIterator implements DataSetIterator {
    private RecordReader recordReader;
    private WritableConverter converter;
    private int batchSize = 10;
    private int labelIndex = -1;
    private int numPossibleLabels = -1;

    public RecordReaderDataSetIterator(RecordReader recordReader, WritableConverter converter, int batchSize,int labelIndex,int numPossibleLabels) {
        this.recordReader = recordReader;
        this.converter = converter;
        this.batchSize = batchSize;
        this.labelIndex = labelIndex;
        this.numPossibleLabels = numPossibleLabels;
    }

    public RecordReaderDataSetIterator(RecordReader recordReader,WritableConverter converter) {
        this(recordReader, converter, 10,-1,-1);
    }


    public RecordReaderDataSetIterator(RecordReader recordReader,WritableConverter converter,int labelIndex,int numPossibleLabels) {
        this(recordReader, converter, 10,labelIndex,numPossibleLabels);
    }


    @Override
    public DataSet next(int num) {
        List<DataSet> dataSets = new ArrayList<>();
        for(int i = 0; i < num; i++) {
            Collection<Writable> record = recordReader.next();
            List<Writable> currList;
            if(record instanceof List)
                currList = (List<Writable>) record;
            else
                currList = new ArrayList<>(record);

            INDArray label = null;
            INDArray featureVector = Nd4j.create(labelIndex >= 0 ? currList.size() -1 : currList.size());
            int count = 0;
            for(int j = 0; j < currList.size(); j++) {
                if(labelIndex >= 0 && j == labelIndex) {
                    if(numPossibleLabels < 1)
                        throw new IllegalStateException("Number of possible labels invalid, must be >= 1");
                    Writable current = currList.get(j);
                    if(converter != null)
                        current = converter.convert(current);
                    label = FeatureUtil.toOutcomeVector(j,Integer.valueOf(current.toString()));
                }
                else {
                    Writable current = currList.get(j);
                    featureVector.putScalar(count++,Double.valueOf(current.toString()));
                }
            }

            dataSets.add(new DataSet(featureVector,labelIndex >= 0 ? label : featureVector));


        }

        List<INDArray> inputs = new ArrayList<>();
        List<INDArray> labels = new ArrayList<>();
        for(DataSet data : dataSets) {
            inputs.add(data.getFeatureMatrix());
            labels.add(data.getLabels());
        }

        return new DataSet(Nd4j.hstack(inputs.toArray(new INDArray[0])),Nd4j.hstack(labels.toArray(new INDArray[0])));
    }

    @Override
    public int totalExamples() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int inputColumns() {
        throw new UnsupportedOperationException();

    }

    @Override
    public int totalOutcomes() {
        throw new UnsupportedOperationException();

    }

    @Override
    public void reset() {

    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        throw new UnsupportedOperationException();

    }

    @Override
    public int numExamples() {
        throw new UnsupportedOperationException();

    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {

    }

    @Override
    public boolean hasNext() {
        return recordReader.hasNext();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();

    }
}
