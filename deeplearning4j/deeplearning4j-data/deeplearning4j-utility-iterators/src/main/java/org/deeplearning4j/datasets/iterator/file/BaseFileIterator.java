/*******************************************************************************
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
 ******************************************************************************/

package org.deeplearning4j.datasets.iterator.file;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.collection.CompactHeapStringList;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.MathUtils;

import java.io.File;
import java.util.*;

/**
 * Base class for loading DataSet objects from file
 *
 * @param <T> Type of dataset
 * @param <P> Type of preprocessor
 * @author Alex Black
 */
public abstract class BaseFileIterator<T, P> implements Iterator<T> {

    protected final List<String> list;
    protected final int batchSize;
    protected final Random rng;

    protected int[] order;
    protected int position;

    private T partialStored;
    @Getter
    @Setter
    protected P preProcessor;


    protected BaseFileIterator(@NonNull File rootDir, int batchSize, String... validExtensions) {
        this(new File[]{rootDir}, true, new Random(), batchSize, validExtensions);
    }

    protected BaseFileIterator(@NonNull File[] rootDirs, boolean recursive, Random rng, int batchSize, String... validExtensions) {
        this.batchSize = batchSize;
        this.rng = rng;

        list = new CompactHeapStringList();
        for(File rootDir : rootDirs) {
            Collection<File> c = FileUtils.listFiles(rootDir, validExtensions, recursive);
            if (c.isEmpty()) {
                throw new IllegalStateException("Root directory is empty (no files found) " + (validExtensions != null ? " (or all files rejected by extension filter)" : ""));
            }
            for (File f : c) {
                list.add(f.getPath());
            }
        }

        if (rng != null) {
            order = new int[list.size()];
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            MathUtils.shuffleArray(order, rng);
        }
    }

    @Override
    public boolean hasNext() {
        return partialStored != null || position < list.size();
    }

    @Override
    public T next() {
        if (!hasNext()) {
            throw new NoSuchElementException("No next element");
        }

        T next;
        if (partialStored != null) {
            next = partialStored;
            partialStored = null;
        } else {
            int nextIdx = (order != null ? order[position++] : position++);
            next = load(new File(list.get(nextIdx)));
        }
        if (batchSize <= 0) {
            //Don't recombine, return as-is
            return next;
        }

        if (sizeOf(next) == batchSize) {
            return next;
        }

        int exampleCount = 0;
        List<T> toMerge = new ArrayList<>();
        toMerge.add(next);
        exampleCount += sizeOf(next);

        while (exampleCount < batchSize && hasNext()) {
            int nextIdx = (order != null ? order[position++] : position++);
            next = load(new File(list.get(nextIdx)));
            exampleCount += sizeOf(next);
            toMerge.add(next);
        }

        T ret = mergeAndStoreRemainder(toMerge);
        applyPreprocessor(ret);
        return ret;
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Not supported");
    }

    protected T mergeAndStoreRemainder(List<T> toMerge) {
        //Could be smaller or larger
        List<T> correctNum = new ArrayList<>();
        List<T> remainder = new ArrayList<>();
        int soFar = 0;
        for (T t : toMerge) {
            int size = sizeOf(t);

            if (soFar + size <= batchSize) {
                correctNum.add(t);
                soFar += size;
            } else if (soFar < batchSize) {
                //Split and add some
                List<T> split = split(t);
                if (rng != null) {
                    Collections.shuffle(split, rng);
                }
                for (T t2 : split) {
                    if (soFar < batchSize) {
                        correctNum.add(t2);
                        soFar += sizeOf(t2);
                    } else {
                        remainder.add(t2);
                    }
                }
            } else {
                //Don't need any of this
                remainder.add(t);
            }
        }

        T ret = merge(correctNum);
        if (remainder.isEmpty()) {
            this.partialStored = null;
        } else {
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                this.partialStored = merge(remainder);
            }
        }

        return ret;
    }


    public void reset() {
        position = 0;
        if (rng != null) {
            MathUtils.shuffleArray(order, rng);
        }
    }

    public boolean resetSupported() {
        return true;
    }

    public boolean asyncSupported() {
        return true;
    }


    protected abstract T load(File f);

    protected abstract int sizeOf(T of);

    protected abstract List<T> split(T toSplit);

    protected abstract T merge(List<T> toMerge);

    protected abstract void applyPreprocessor(T toPreProcess);
}
