/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.dataset.adapter;

import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * Iterator that adapts a DataSetIterator to a MultiDataSetIterator
 *
 * @author Alex Black
 */
public class MultiDataSetIteratorAdapter implements MultiDataSetIterator {

    private org.nd4j.linalg.dataset.api.iterator.DataSetIterator iter;
    private MultiDataSetPreProcessor preProcessor;

    public MultiDataSetIteratorAdapter(org.nd4j.linalg.dataset.api.iterator.DataSetIterator iter) {
        this.iter = iter;
    }

    @Override
    public MultiDataSet next(int i) {
        MultiDataSet mds = iter.next(i).toMultiDataSet();
        if (preProcessor != null)
            preProcessor.preProcess(mds);
        return mds;
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor multiDataSetPreProcessor) {
        this.preProcessor = multiDataSetPreProcessor;
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return preProcessor;
    }

    @Override
    public boolean resetSupported() {
        return iter.resetSupported();
    }

    @Override
    public boolean asyncSupported() {
        return iter.asyncSupported();
    }

    @Override
    public void reset() {
        iter.reset();
    }

    @Override
    public boolean hasNext() {
        return iter.hasNext();
    }

    @Override
    public MultiDataSet next() {
        MultiDataSet mds = iter.next().toMultiDataSet();
        if (preProcessor != null)
            preProcessor.preProcess(mds);
        return mds;
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

}
