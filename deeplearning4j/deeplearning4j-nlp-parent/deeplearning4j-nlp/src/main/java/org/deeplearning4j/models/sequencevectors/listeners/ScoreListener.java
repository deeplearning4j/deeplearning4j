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

package org.deeplearning4j.models.sequencevectors.listeners;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.enums.ListenerEvent;
import org.deeplearning4j.models.sequencevectors.interfaces.VectorsListener;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Simple VectorsListener implementation that prints out model score.
 *
 * PLEASE NOTE: THIS IS PLACEHOLDER FOR FUTURE IMPLEMENTATION
 *
 * @author raver119@gmail.com
 */
@Deprecated
public class ScoreListener<T extends SequenceElement> implements VectorsListener<T> {
    protected static final Logger logger = LoggerFactory.getLogger(ScoreListener.class);
    private final ListenerEvent targetEvent;
    private final AtomicLong callsCount = new AtomicLong(0);
    private final int frequency;

    public ScoreListener(@NonNull ListenerEvent targetEvent, int frequency) {
        this.targetEvent = targetEvent;
        this.frequency = frequency;
    }

    @Override
    public boolean validateEvent(ListenerEvent event, long argument) {
        if (event == targetEvent)
            return true;

        return false;
    }

    @Override
    public void processEvent(ListenerEvent event, SequenceVectors<T> sequenceVectors, long argument) {
        if (event != targetEvent)
            return;

        callsCount.incrementAndGet();

        if (callsCount.get() % frequency == 0)
            logger.info("Average score for last batch: {}", sequenceVectors.getElementsScore());
    }
}
