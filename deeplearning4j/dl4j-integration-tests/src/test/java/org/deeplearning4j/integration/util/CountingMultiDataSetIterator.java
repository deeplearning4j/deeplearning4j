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

package org.deeplearning4j.integration.util;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * A simple iterator that counts the expected number of parameter updates.
 * Accounts for TBPTT (i.e., multiple updates per MultiDataSet) if used
 *
 * @author Alex Black
 */
@Data
public class CountingMultiDataSetIterator implements MultiDataSetIterator {

    private MultiDataSetIterator underlying;
    private int currIter = 0;
    private IntArrayList iterAtReset = new IntArrayList();
    private boolean tbptt;
    private int tbpttLength;

    /**
     *
     * @param underlying  Underlying iterator
     * @param tbptt       Whether TBPTT is used
     * @param tbpttLength Network TBPTT length
     */
    public CountingMultiDataSetIterator(MultiDataSetIterator underlying, boolean tbptt, int tbpttLength){
        this.underlying = underlying;
        this.tbptt = tbptt;
        this.tbpttLength = tbpttLength;
    }

    @Override
    public MultiDataSet next(int i) {
        currIter++;
        return underlying.next(i);
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor multiDataSetPreProcessor) {
        underlying.setPreProcessor(multiDataSetPreProcessor);
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return underlying.getPreProcessor();
    }

    @Override
    public boolean resetSupported() {
        return underlying.resetSupported();
    }

    @Override
    public boolean asyncSupported() {
        return underlying.asyncSupported();
    }

    @Override
    public void reset() {
        underlying.reset();
        iterAtReset.add(currIter);
        currIter = 0;
    }

    @Override
    public boolean hasNext() {
        return underlying.hasNext();
    }

    @Override
    public MultiDataSet next() {
        MultiDataSet mds = underlying.next();
        if(tbptt){
            INDArray f = mds.getFeatures(0);
            if(f.rank() == 3){
                int numSegments = (int)Math.ceil(f.size(2) / (double)tbpttLength);
                currIter += numSegments;
            }
        } else {
            currIter++;
        }
        return mds;
    }
}
