package org.deeplearning4j.datasets.iterator.loader;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.api.loader.Loader;
import org.nd4j.api.loader.Source;
import org.nd4j.api.loader.SourceFactory;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.util.MathUtils;

import java.io.IOException;
import java.util.*;

/**
 * A MultiDataSetLoader that loads MultiDataSets from a path, using a {@code Loader<MultiDataSet>} such as {@link SerializedMultiDataSetLoader}.
 * Paths are converted to input streams using {@link SourceFactory} such as {@link org.nd4j.api.loader.LocalFileSourceFactory}.
 * Note that this iterator does not implement any sort of merging/batching functionality - it simply returns the DataSets
 * as-is from the path/loader.
 *
 * Note: If using {@link #MultiDataSetLoaderIterator(Collection, Random, Loader, SourceFactory)} constructor with non-null
 * Random instance, the data will be shuffled,
 *
 *
 * @author Alex Black
 */
@Data
public class MultiDataSetLoaderIterator implements MultiDataSetIterator {

    protected final List<String> paths;
    protected final Iterator<String> iter;
    protected final Loader<MultiDataSet> loader;
    protected final SourceFactory sourceFactory;
    protected final Random rng;
    protected final int[] order;
    protected MultiDataSetPreProcessor preProcessor;
    protected int position;

    /**
     * NOTE: When using this constructor (with {@code Iterator<String>}) the MultiDataSetIterator cannot be reset.
     * Use the other construtor that takes {@code Collection<String>}
     *
     * @param paths         Paths to iterate over
     * @param loader        Loader to use when loading DataSets
     * @param sourceFactory The factory to use to convert the paths into streams via {@link Source}
     */
    public MultiDataSetLoaderIterator(Iterator<String> paths, Loader<MultiDataSet> loader, SourceFactory sourceFactory){
        this.paths = null;
        this.iter = paths;
        this.loader = loader;
        this.sourceFactory = sourceFactory;
        this.rng = null;
        this.order = null;
    }

    /**
     * Iterate of the specified collection of strings without randomization
     *
     * @param paths         Paths to iterate over
     * @param loader        Loader to use when loading DataSets
     * @param sourceFactory The factory to use to convert the paths into streams via {@link Source}
     */
    public MultiDataSetLoaderIterator(Collection<String> paths, Loader<MultiDataSet> loader, SourceFactory sourceFactory) {
        this(paths, null, loader, sourceFactory);
    }

    /**
     * Iterate of the specified collection of strings with optional randomization
     *
     * @param paths         Paths to iterate over
     * @param rng           Optional random instance to use for shuffling of order. If null, no shuffling will be used.
     * @param loader        Loader to use when loading DataSets
     * @param sourceFactory The factory to use to convert the paths into streams via {@link Source}
     */
    public MultiDataSetLoaderIterator(Collection<String> paths, Random rng, Loader<MultiDataSet> loader, SourceFactory sourceFactory) {
        if(paths instanceof List){
            this.paths = (List<String>)paths;
        } else {
            this.paths = new ArrayList<>(paths);
        }
        this.rng = rng;
        this.loader = loader;
        this.sourceFactory = sourceFactory;
        this.iter = null;

        if(rng != null){
            order = new int[paths.size()];
            for( int i=0; i<order.length; i++ ){
                order[i] = i;
            }
            MathUtils.shuffleArray(order, rng);
        } else {
            order = null;
        }
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
        position = 0;
        if (rng != null) {
            MathUtils.shuffleArray(order, rng);
        }
    }

    @Override
    public boolean hasNext() {
        if(iter != null)
            return iter.hasNext();
        return position < paths.size();
    }

    @Override
    public MultiDataSet next() {
        if(!hasNext())
            throw new NoSuchElementException("No next element");
        String path;
        if(iter != null){
            path = iter.next();
        } else {
            if(order != null){
                path = paths.get(order[position++]);
            } else {
                path = paths.get(position++);
            }
        }
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
