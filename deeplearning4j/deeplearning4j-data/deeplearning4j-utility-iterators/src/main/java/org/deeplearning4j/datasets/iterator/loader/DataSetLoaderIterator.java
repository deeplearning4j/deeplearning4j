package org.deeplearning4j.datasets.iterator.loader;

import lombok.Getter;
import lombok.Setter;
import org.nd4j.api.loader.Loader;
import org.nd4j.api.loader.Source;
import org.nd4j.api.loader.SourceFactory;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

public class DataSetLoaderIterator implements DataSetIterator {

    protected final Iterable<String> paths;
    protected final Loader<DataSet> loader;

    protected Iterator<String> iter;
    @Getter @Setter
    protected DataSetPreProcessor preProcessor;
    protected SourceFactory sourceFactory;

    public DataSetLoaderIterator(Iterator<String> paths, Loader<DataSet> loader, SourceFactory sourceFactory){
        this.paths = null;
        this.iter = paths;
        this.loader = loader;
        this.sourceFactory = sourceFactory;
    }

    public DataSetLoaderIterator(Iterable<String> paths, Loader<DataSet> loader, SourceFactory sourceFactory){
        this.paths = paths;
        this.loader = loader;
        this.sourceFactory = sourceFactory;
        this.iter = paths.iterator();
    }

    @Override
    public DataSet next(int i) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public int inputColumns() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public int totalOutcomes() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public boolean resetSupported() {
        return paths != null;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        if(!resetSupported())
             throw new UnsupportedOperationException("Reset not supported when using Iterator<String> instead of Iterable<String>");
        this.iter = paths.iterator();
    }

    @Override
    public int batch() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public boolean hasNext() {
        return iter.hasNext();
    }

    @Override
    public DataSet next() {
        if(!hasNext())
            throw new NoSuchElementException("No next element");
        String path = iter.next();
        Source s = sourceFactory.getSource(path);
        DataSet ds;
        try {
            ds = loader.load(s);
        } catch (IOException e){
            throw new RuntimeException(e);
        }
        if(preProcessor != null)
            preProcessor.preProcess(ds);
        return ds;
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Not supported");
    }
}
