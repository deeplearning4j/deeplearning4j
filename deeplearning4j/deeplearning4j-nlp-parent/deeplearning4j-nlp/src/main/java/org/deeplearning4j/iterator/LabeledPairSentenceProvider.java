/*
 *  ******************************************************************************
 *  *
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

package org.deeplearning4j.iterator;

import org.nd4j.common.primitives.Triple;

import java.util.List;

/**
 * LabeledPairSentenceProvider: a simple iterator interface over a pair of sentences/documents that have a label.<br>
 */
public interface LabeledPairSentenceProvider {

    /**
     * Are there more sentences/documents available?
     */
    boolean hasNext();

    /**
     * @return Triple: two sentence/document texts and label
     */
    Triple<String, String, String> nextSentencePair();

    /**
     * Reset the iterator - including shuffling the order, if necessary/appropriate
     */
    void reset();

    /**
     * Return the total number of sentences, or -1 if not available
     */
    int totalNumSentences();

    /**
     * Return the list of labels - this also defines the class/integer label assignment order
     */
    List<String> allLabels();

    /**
     * Equivalent to allLabels().size()
     */
    int numLabelClasses();

}


