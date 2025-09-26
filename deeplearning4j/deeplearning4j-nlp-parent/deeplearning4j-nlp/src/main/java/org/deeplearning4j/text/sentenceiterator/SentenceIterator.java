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

package org.deeplearning4j.text.sentenceiterator;

public interface SentenceIterator {

    /**
     * Gets the next sentence or null
     * if there's nothing left (Do yourself a favor and
     * check hasNext() )
     * 
     * @return the next sentence in the iterator
     */
    String nextSentence();

    /**
     * Same idea as {@link java.util.Iterator}
     * @return whether there's anymore sentences left
     */
    boolean hasNext();

    /**
     * Resets the iterator to the beginning
     */
    void reset();

    /**
     * Allows for any finishing (closing of input streams or the like)
     */
    void finish();


    SentencePreProcessor getPreProcessor();

    void setPreProcessor(SentencePreProcessor preProcessor);


}
