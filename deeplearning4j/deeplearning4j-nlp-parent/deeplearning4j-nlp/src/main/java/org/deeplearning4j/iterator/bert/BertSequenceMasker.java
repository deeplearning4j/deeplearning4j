/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.iterator.bert;

import org.nd4j.common.primitives.Pair;

import java.util.List;

/**
 * Interface used to customize how masking should be performed with {@link org.deeplearning4j.iterator.BertIterator}
 * when doing unsupervised training
 *
 * @author Alex Black
 */
public interface BertSequenceMasker {

    /**
     *
     * @param input         Input sequence of tokens
     * @param maskToken     Token to use for masking - usually something like "[MASK]"
     * @param vocabWords    Vocabulary, as a list
     * @return Pair: The new input tokens (after masking out), along with a boolean[] for whether the token is
     * masked or not (same length as number of tokens). boolean[i] is true if token i was masked.
     */
    Pair<List<String>,boolean[]> maskSequence(List<String> input, String maskToken, List<String> vocabWords);

}
