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

package org.deeplearning4j.spark.models.sequencevectors.functions;

import lombok.NonNull;
import org.apache.spark.Accumulator;
import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.spark.models.sequencevectors.primitives.ExtraCounter;
import org.nd4j.linalg.primitives.Pair;

/**
 * This accumulator function does count individual elements, using provided Accumulator
 *
 * @author raver119@gmail.com
 */
public class ExtraCountFunction<T extends SequenceElement> implements Function<Sequence<T>, Pair<Sequence<T>, Long>> {
    protected Accumulator<ExtraCounter<Long>> accumulator;
    protected boolean fetchLabels;

    public ExtraCountFunction(@NonNull Accumulator<ExtraCounter<Long>> accumulator, boolean fetchLabels) {
        this.accumulator = accumulator;
        this.fetchLabels = fetchLabels;
    }

    @Override
    public Pair<Sequence<T>, Long> call(Sequence<T> sequence) throws Exception {
        // since we can't be 100% sure that sequence size is ok itself, or it's not overflow through int limits, we'll recalculate it.
        // anyway we're going to loop through it for elements frequencies
        ExtraCounter<Long> localCounter = new ExtraCounter<>();
        long seqLen = 0;

        for (T element : sequence.getElements()) {
            if (element == null)
                continue;

            // FIXME: hashcode is bad idea here. we need Long id
            localCounter.incrementCount(element.getStorageId(), 1.0f);
            seqLen++;
        }

        // FIXME: we're missing label information here due to shallow vocab mechanics
        if (sequence.getSequenceLabels() != null)
            for (T label : sequence.getSequenceLabels()) {
                localCounter.incrementCount(label.getStorageId(), 1.0f);
            }

        localCounter.buildNetworkSnapshot();

        accumulator.add(localCounter);

        return Pair.makePair(sequence, seqLen);
    }
}
