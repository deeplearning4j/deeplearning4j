/*
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package org.deeplearning4j.datasets.iterator.loader;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.api.loader.Loader;
import org.nd4j.api.loader.Source;
import org.nd4j.api.loader.SourceFactory;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.util.MathUtils;

import java.io.IOException;
import java.util.*;

/**
 * A DataSetLoader that loads DataSets from a path, using a {@code Loader<DataSet>} such as SerializedDataSetLoader.
 * Paths are converted to input streams using {@link SourceFactory} such as {@link org.nd4j.api.loader.LocalFileSourceFactory}.
 * Note that this iterator does not implement any sort of merging/batching functionality - it simply returns the DataSets
 * as-is from the path/loader.
 *
 * Note: If using {@link #DataSetLoaderIterator(Collection, Random, Loader, SourceFactory)} constructor with non-null
 * Random instance, the data will be shuffled,
 *
 *
 * @author Alex Black
 */
@Data
public class DataSetLoaderIterator implements DataSetIterator {

    protected final List<String> paths;
    protected final Iterator<String> iter;
    protected final SourceFactory sourceFactory;
    protected final Loader<DataSet> loader;
    protected final Random rng;
    protected final int[] order;
    protected int position;

    @Getter @Setter
    protected DataSetPreProcessor preProcessor;

    /**
     * NOTE: When using this constructor (with {@code Iterator<String>}) the DataSetIterator cannot be reset.
     * Use the other construtor that takes {@code Collection<String>}
     *
     * @param paths         Paths to iterate over
     * @param loader        Loader to use when loading DataSets
     * @param sourceFactory The factory to use to convert the paths into streams via {@link Source}
     */
    public DataSetLoaderIterator(Iterator<String> paths, Loader<DataSet> loader, SourceFactory sourceFactory){
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
    public DataSetLoaderIterator(Collection<String> paths, Loader<DataSet> loader, SourceFactory sourceFactory) {
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
    public DataSetLoaderIterator(Collection<String> paths, Random rng, Loader<DataSet> loader, SourceFactory sourceFactory){
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
        position = 0;
        if (rng != null) {
            MathUtils.shuffleArray(order, rng);
        }
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
        if(iter != null)
            return iter.hasNext();
        return position < paths.size();
    }

    @Override
    public DataSet next() {
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
