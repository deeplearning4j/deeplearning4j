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

package org.deeplearning4j.models.sequencevectors.interfaces;

import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.enums.ListenerEvent;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * This interface describes Listeners to SequenceVectors and its derivatives.
 *
 * @author raver119@gmail.com
 */
public interface VectorsListener<T extends SequenceElement> {

    /**
     * This method is called prior each processEvent call, to check if this specific VectorsListener implementation is viable for specific event
     *
     * @param event
     * @param argument
     * @return TRUE, if this event can and should be processed with this listener, FALSE otherwise
     */
    boolean validateEvent(ListenerEvent event, long argument);

    /**
     * This method is called at each epoch end
     *
     * @param event
     * @param sequenceVectors
     */
    void processEvent(ListenerEvent event, SequenceVectors<T> sequenceVectors, long argument);
}
