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

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.BlockMultiDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.ArrayList;

/**
 * This class provides baseline implementation of BlockMultiDataSetIterator interface
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class DummyBlockMultiDataSetIterator implements BlockMultiDataSetIterator {
    protected final MultiDataSetIterator iterator;

    public DummyBlockMultiDataSetIterator(@NonNull MultiDataSetIterator iterator) {
        this.iterator = iterator;
    }

    @Override
    public boolean hasAnything() {
        return iterator.hasNext();
    }

    @Override
    public MultiDataSet[] next(int maxDatasets) {
        val list = new ArrayList<MultiDataSet>(maxDatasets);
        int cnt = 0;
        while (iterator.hasNext() && cnt < maxDatasets) {
            list.add(iterator.next());
            cnt++;
        }

        return list.toArray(new MultiDataSet[list.size()]);
    }
}
