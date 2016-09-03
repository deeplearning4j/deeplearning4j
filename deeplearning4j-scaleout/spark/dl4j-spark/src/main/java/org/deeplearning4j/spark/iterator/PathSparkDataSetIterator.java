package org.deeplearning4j.spark.iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;
import java.io.OutputStream;
import java.net.URI;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * A DataSetIterator that loads serialized DataSet objects (saved with {@link DataSet#save(OutputStream)}) from
 * a String that represents the path (for example, on HDFS)
 *
 * @author Alex Black
 */
public class PathSparkDataSetIterator implements DataSetIterator {

    public static final int BUFFER_SIZE = 4194304;  //4 MB

    private final Collection<String> dataSetStreams;
    private DataSetPreProcessor preprocessor;
    private Iterator<String> iter;
    private int totalOutcomes = -1;
    private int inputColumns = -1;
    private int batch = -1;
    private int cursor = 0;
    private DataSet preloadedDataSet;
    private FileSystem fileSystem;

    public PathSparkDataSetIterator(Iterator<String> iter){
        this.dataSetStreams = null;
        this.iter = iter;
    }

    public PathSparkDataSetIterator(Collection<String> dataSetStreams){
        this.dataSetStreams = dataSetStreams;
        iter = dataSetStreams.iterator();
    }

    @Override
    public DataSet next(int num) {
        return next();
    }

    @Override
    public int totalExamples() {
        throw new UnsupportedOperationException("Total examples unknown for PathSparkDataSetIterator");
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

    private synchronized DataSet load(String path){
        if(fileSystem == null){
            try{
                fileSystem = FileSystem.get(new URI(path), new Configuration());
            }catch(Exception e){
                throw new RuntimeException(e);
            }
        }

        DataSet ds = new DataSet();
        try(FSDataInputStream inputStream = fileSystem.open(new Path(path), BUFFER_SIZE)){
            ds.load(inputStream);
        }catch(IOException e){
            throw new RuntimeException(e);
        }

        cursor++;
        return ds;
    }
}
