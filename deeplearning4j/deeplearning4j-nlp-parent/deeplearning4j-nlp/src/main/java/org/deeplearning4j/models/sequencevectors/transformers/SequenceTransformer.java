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

package org.deeplearning4j.models.sequencevectors.transformers;

import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 *
 * @author raver119@gmail.com
 */
public interface SequenceTransformer<T extends SequenceElement, V extends Object> {

    /**
     * Returns Vocabulary derived from underlying data source.
     * In default implementations this method heavily relies on transformToSequence() method.
     *
     * @return
     */
    //VocabCache<T> derivedVocabulary();

    /**
     * This is generic method for transformation data from any format to Sequence of SequenceElement.
     * It will be used both in Vocab building, and in training process
     *
     * @param object - Object to be transformed into Sequence
     * @return
     */
    Sequence<T> transformToSequence(V object);


    void reset();
}
