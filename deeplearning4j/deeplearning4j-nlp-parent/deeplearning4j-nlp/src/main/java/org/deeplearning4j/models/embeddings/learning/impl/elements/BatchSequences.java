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

package org.deeplearning4j.models.embeddings.learning.impl.elements;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

@Slf4j
public class BatchSequences<T extends SequenceElement> {

    private int batches;

    List<BatchItem<T>> buffer = new ArrayList<>();

    public BatchSequences(int batches) {
        this.batches = batches;
    }

    public void put(T word, T lastWord, long randomValue, double alpha) {
        BatchItem<T> newItem = new BatchItem<>(word, lastWord, randomValue, alpha);
        buffer.add(newItem);
    }

    public void put(T word, int[] windowWords, boolean[] wordStatuses, long randomValue, double alpha) {
        BatchItem<T> newItem = new BatchItem<>(word, windowWords, wordStatuses, randomValue, alpha);
        buffer.add(newItem);
    }

    public void put(T word, int[] windowWords, boolean[] wordStatuses, long randomValue, double alpha, int numLabels) {
        BatchItem<T> newItem = new BatchItem<>(word, windowWords, wordStatuses, randomValue, alpha, numLabels);
        buffer.add(newItem);
    }

    public List<BatchItem<T>> get(int chunkNo) {
        List<BatchItem<T>> retVal = new ArrayList<>();

        for (int i = 0 + chunkNo * batches; (i < batches + chunkNo * batches) && (i < buffer.size()); ++i) {
            BatchItem<T> value = buffer.get(i);
            retVal.add(value);
        }
        return retVal;
    }

    public int size() {
        return buffer.size();
    }

    public void clear() {
        buffer.clear();
    }


}
