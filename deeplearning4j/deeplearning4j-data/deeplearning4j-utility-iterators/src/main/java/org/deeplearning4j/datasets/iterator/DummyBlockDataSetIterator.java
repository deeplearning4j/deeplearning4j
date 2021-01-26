/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.datasets.iterator;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.BlockDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.ArrayList;

/**
 * This class provides baseline implementation of BlockDataSetIterator interface
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class DummyBlockDataSetIterator implements BlockDataSetIterator {
    protected final DataSetIterator iterator;

    public DummyBlockDataSetIterator(@NonNull DataSetIterator iterator) {
        this.iterator = iterator;
    }

    @Override
    public boolean hasAnything() {
        return iterator.hasNext();
    }

    @Override
    public DataSet[] next(int maxDatasets) {
        val list = new ArrayList<DataSet>(maxDatasets);
        int cnt = 0;
        while (iterator.hasNext() && cnt < maxDatasets) {
            list.add(iterator.next());
            cnt++;
        }

        return list.toArray(new DataSet[list.size()]);
    }
}
