package org.deeplearning4j.spark.iterator;

import org.apache.spark.input.PortableDataStream;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.OutputStream;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * A DataSetIterator that loads serialized DataSet objects (saved with {@link DataSet#save(OutputStream)}) from
 * a {@link PortableDataStream}, usually obtained from SparkContext.binaryFiles()
 *
 * @author Alex Black
 */
public class PortableDataStreamDataSetIterator implements DataSetIterator {

    private final Collection<PortableDataStream> dataSetStreams;
    private DataSetPreProcessor preprocessor;
    private Iterator<PortableDataStream> iter;
    private int totalOutcomes = -1;
    private int inputColumns = -1;
    private int batch = -1;
    private int cursor = 0;
    private DataSet preloadedDataSet;

    public PortableDataStreamDataSetIterator(Iterator<PortableDataStream> iter){
        this.dataSetStreams = null;
        this.iter = iter;
    }

    public PortableDataStreamDataSetIterator(Collection<PortableDataStream> dataSetStreams){
        this.dataSetStreams = dataSetStreams;
        iter = dataSetStreams.iterator();
    }

    @Override
    public DataSet next(int num) {
        return next();
    }

    @Override
    public int totalExamples() {
        throw new UnsupportedOperationException("Total examples unknown for PortableDataStreamDataSetIterator");
    }

    @Override
    public int inputColumns() {
        if(inputColumns == -1) preloadDataSet();
        return inputColumns;
    }

    @Override
    public int totalOutcomes() {
        if(totalOutcomes == -1) preloadDataSet();
        return totalExamples();
    }

    @Override
    public boolean resetSupported(){
        return dataSetStreams != null;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        if(dataSetStreams == null) throw new IllegalStateException("Cannot reset iterator constructed with an iterator");
        iter = dataSetStreams.iterator();
        cursor = 0;
    }

    @Override
    public int batch() {
        if(batch == -1) preloadDataSet();
        return batch;
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        return 0;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.preprocessor = preProcessor;
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return this.preprocessor;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return iter.hasNext();
    }

    @Override
    public DataSet next() {
        DataSet ds;
        if(preloadedDataSet != null){
            ds = preloadedDataSet;
            preloadedDataSet = null;
        } else {
            ds = load(iter.next());
        }

        totalOutcomes = ds.getLabels().size(1);
        inputColumns = ds.getFeatureMatrix().size(1);
        batch = ds.numExamples();

        if(preprocessor != null) preprocessor.preProcess(ds);
        return ds;
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

    private void preloadDataSet(){
        preloadedDataSet = load(iter.next());
        totalOutcomes = preloadedDataSet.getLabels().size(1);
        inputColumns = preloadedDataSet.getFeatureMatrix().size(1);
        batch = preloadedDataSet.numExamples();
    }

    private DataSet load(PortableDataStream pds){
        DataSet ds = new DataSet();
        try{
            ds.load(pds.open());
        } finally {
            pds.close();
        }
        cursor++;
        return ds;
    }
}
