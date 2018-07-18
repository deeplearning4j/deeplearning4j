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

package org.deeplearning4j.datasets.iterator;


import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

/**
 * A DataSetIterator that works on an Iterator<DataSet>, combining and splitting the input DataSet objects as
 * required to get a consistent batch size.
 *
 * Typically used in Spark training, but may be used elsewhere.
 * NOTE: reset method is not supported here.
 */
public class IteratorMultiDataSetIterator implements MultiDataSetIterator {

    private final Iterator<MultiDataSet> iterator;
    private final int batchSize;
    private final LinkedList<MultiDataSet> queued; //Used when splitting larger examples than we want to return in a batch
    private MultiDataSetPreProcessor preProcessor;

    public IteratorMultiDataSetIterator(Iterator<MultiDataSet> iterator, int batchSize) {
        this.iterator = iterator;
        this.batchSize = batchSize;
        this.queued = new LinkedList<>();
    }

    @Override
    public boolean hasNext() {
        return !queued.isEmpty() || iterator.hasNext();
    }

    @Override
    public MultiDataSet next() {
        return next(batchSize);
    }

    @Override
    public MultiDataSet next(int num) {
        if (!hasNext())
            throw new NoSuchElementException();

        List<MultiDataSet> list = new ArrayList<>();
        int countSoFar = 0;
        while ((!queued.isEmpty() || iterator.hasNext()) && countSoFar < batchSize) {
            MultiDataSet next;
            if (!queued.isEmpty()) {
                next = queued.removeFirst();
            } else {
                next = iterator.next();
            }

            // FIXME: int cast
            int nExamples = (int) next.getFeatures(0).size(0);
            if (countSoFar + nExamples <= batchSize) {
                //Add the entire MultiDataSet as-is
                list.add(next);
            } else {
                //Split the MultiDataSet

                int nFeatures = next.numFeatureArrays();
                int nLabels = next.numLabelsArrays();

                INDArray[] fToKeep = new INDArray[nFeatures];
                INDArray[] lToKeep = new INDArray[nLabels];
                INDArray[] fToCache = new INDArray[nFeatures];
                INDArray[] lToCache = new INDArray[nLabels];
                INDArray[] fMaskToKeep = (next.getFeaturesMaskArrays() != null ? new INDArray[nFeatures] : null);
                INDArray[] lMaskToKeep = (next.getLabelsMaskArrays() != null ? new INDArray[nLabels] : null);
                INDArray[] fMaskToCache = (next.getFeaturesMaskArrays() != null ? new INDArray[nFeatures] : null);
                INDArray[] lMaskToCache = (next.getLabelsMaskArrays() != null ? new INDArray[nLabels] : null);

                for (int i = 0; i < nFeatures; i++) {
                    INDArray fi = next.getFeatures(i);
                    fToKeep[i] = getRange(fi, 0, batchSize - countSoFar);
                    fToCache[i] = getRange(fi, batchSize - countSoFar, nExamples);

                    if (fMaskToKeep != null) {
                        INDArray fmi = next.getFeaturesMaskArray(i);
                        fMaskToKeep[i] = getRange(fmi, 0, batchSize - countSoFar);
                        fMaskToCache[i] = getRange(fmi, batchSize - countSoFar, nExamples);
                    }
                }

                for (int i = 0; i < nLabels; i++) {
                    INDArray li = next.getLabels(i);
                    lToKeep[i] = getRange(li, 0, batchSize - countSoFar);
                    lToCache[i] = getRange(li, batchSize - countSoFar, nExamples);

                    if (lMaskToKeep != null) {
                        INDArray lmi = next.getLabelsMaskArray(i);
                        lMaskToKeep[i] = getRange(lmi, 0, batchSize - countSoFar);
                        lMaskToCache[i] = getRange(lmi, batchSize - countSoFar, nExamples);
                    }
                }

                MultiDataSet toKeep =
                                new org.nd4j.linalg.dataset.MultiDataSet(fToKeep, lToKeep, fMaskToKeep, lMaskToKeep);
                MultiDataSet toCache = new org.nd4j.linalg.dataset.MultiDataSet(fToCache, lToCache, fMaskToCache,
                                lMaskToCache);
                list.add(toKeep);
                queued.add(toCache);
            }

            countSoFar += nExamples;
        }

        MultiDataSet out;
        if (list.size() == 1) {
            out = list.get(0);
        } else {
            out = org.nd4j.linalg.dataset.MultiDataSet.merge(list);
        }

        if (preProcessor != null)
            preProcessor.preProcess(out);
        return out;
    }

    private static INDArray getRange(INDArray arr, int exampleFrom, int exampleToExclusive) {
        if (arr == null)
            return null;

        int rank = arr.rank();
        switch (rank) {
            case 2:
                return arr.get(NDArrayIndex.interval(exampleFrom, exampleToExclusive), NDArrayIndex.all());
            case 3:
                return arr.get(NDArrayIndex.interval(exampleFrom, exampleToExclusive), NDArrayIndex.all(),
                                NDArrayIndex.all());
            case 4:
                return arr.get(NDArrayIndex.interval(exampleFrom, exampleToExclusive), NDArrayIndex.all(),
                                NDArrayIndex.all(), NDArrayIndex.all());
            default:
                throw new RuntimeException("Invalid rank: " + rank);
        }
    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        //No need to asynchronously prefetch here: already in memory
        return false;
    }

    @Override
    public void reset() {
        throw new UnsupportedOperationException("Reset not supported");
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Not supported");
    }
}
