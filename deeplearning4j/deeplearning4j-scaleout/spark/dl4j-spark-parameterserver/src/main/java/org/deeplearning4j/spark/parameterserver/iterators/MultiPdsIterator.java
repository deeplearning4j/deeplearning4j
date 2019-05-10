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

package org.deeplearning4j.spark.parameterserver.iterators;

import lombok.NonNull;
import org.apache.spark.input.PortableDataStream;
import org.deeplearning4j.spark.parameterserver.callbacks.PortableDataStreamMDSCallback;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.util.Iterator;
import java.util.function.Consumer;

/**
 * @author raver119@gmail.com
 */
public class MultiPdsIterator implements Iterator<MultiDataSet> {
    protected final Iterator<PortableDataStream> iterator;
    protected final PortableDataStreamMDSCallback callback;

    public MultiPdsIterator(@NonNull Iterator<PortableDataStream> pds,
                    @NonNull PortableDataStreamMDSCallback callback) {
        this.iterator = pds;
        this.callback = callback;
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public MultiDataSet next() {
        return callback.compute(iterator.next());
    }

    @Override
    public void remove() {
        // no-op
    }

    @Override
    public void forEachRemaining(Consumer<? super MultiDataSet> action) {
        throw new UnsupportedOperationException();
    }
}
