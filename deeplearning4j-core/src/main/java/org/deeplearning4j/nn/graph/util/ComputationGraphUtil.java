/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.graph.util;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

public class ComputationGraphUtil {

    /** Convert a DataSet to the equivalent MultiDataSet */
    public static MultiDataSet toMultiDataSet(DataSet dataSet){
        INDArray f = dataSet.getFeatureMatrix();
        INDArray l = dataSet.getLabels();
        INDArray fMask = dataSet.getFeaturesMaskArray();
        INDArray lMask = dataSet.getLabelsMaskArray();

        INDArray[] fNew = new INDArray[]{f};
        INDArray[] lNew = new INDArray[]{l};
        INDArray[] fMaskNew = (fMask != null ? new INDArray[]{fMask} : null);
        INDArray[] lMaskNew = (lMask != null ? new INDArray[]{lMask} : null);

        return new org.nd4j.linalg.dataset.MultiDataSet(fNew,lNew,fMaskNew,lMaskNew);
    }

    /** Convert a DataSetIterator to a MultiDataSetIterator, via an adaptor class */
    public static MultiDataSetIterator toMultiDataSetIterator(DataSetIterator iterator){
        return new MultiDataSetIteratorAdaptor(iterator);
    }


    private static class MultiDataSetIteratorAdaptor implements MultiDataSetIterator {

        private DataSetIterator iter;
        private MultiDataSetPreProcessor preProcessor;

        public MultiDataSetIteratorAdaptor(DataSetIterator iter) {
            this.iter = iter;
        }

        @Override
        public MultiDataSet next(int i) {
            MultiDataSet mds = toMultiDataSet(iter.next(i));
            if(preProcessor != null) preProcessor.preProcess(mds);
            return mds;
        }

        @Override
        public void setPreProcessor(MultiDataSetPreProcessor multiDataSetPreProcessor) {
            this.preProcessor = multiDataSetPreProcessor;
        }

        @Override
        public void reset() {
            iter.reset();;
        }

        @Override
        public boolean hasNext() {
            return iter.hasNext();
        }

        @Override
        public MultiDataSet next() {
            MultiDataSet mds = toMultiDataSet(iter.next());
            if(preProcessor != null) preProcessor.preProcess(mds);
            return mds;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }

}
