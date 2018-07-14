package org.deeplearning4j.datasets.iterator.loader;

import lombok.Getter;
import lombok.Setter;
import org.nd4j.api.loader.Loader;
import org.nd4j.api.loader.Source;
import org.nd4j.api.loader.SourceFactory;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

public class MultiDataSetLoaderIterator implements MultiDataSetIterator {

    protected final Iterable<String> paths;
    protected final Loader<MultiDataSet> loader;

    protected Iterator<String> iter;
    @Getter @Setter
    protected MultiDataSetPreProcessor preProcessor;
    protected SourceFactory sourceFactory;

    public MultiDataSetLoaderIterator(Iterator<String> paths, Loader<MultiDataSet> loader, SourceFactory sourceFactory){
        this.paths = null;
        this.iter = paths;
        this.loader = loader;
        this.sourceFactory = sourceFactory;
    }

    public MultiDataSetLoaderIterator(Iterable<String> paths, Loader<MultiDataSet> loader, SourceFactory sourceFactory){
        this.paths = paths;
        this.loader = loader;
        this.sourceFactory = sourceFactory;
        this.iter = paths.iterator();
    }

    @Override
    public MultiDataSet next(int i) {
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
    public boolean hasNext() {
        return iter.hasNext();
    }

    @Override
    public MultiDataSet next() {
        if(!hasNext())
            throw new NoSuchElementException("No next element");
        String path = iter.next();
        Source s = sourceFactory.getSource(path);
        MultiDataSet mds;
        try {
            mds = loader.load(s);
        } catch (IOException e){
            throw new RuntimeException(e);
        }
        if(preProcessor != null)
            preProcessor.preProcess(mds);
        return mds;
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Not supported");
    }
}
