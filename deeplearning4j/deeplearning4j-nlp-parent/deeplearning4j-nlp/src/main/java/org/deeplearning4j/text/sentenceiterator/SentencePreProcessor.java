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

package org.deeplearning4j.text.sentenceiterator;

import java.io.Serializable;

/**
 * Sentence pre processor.
 * Used for pre processing strings
 *
 * @author Adam Gibson
 */
public interface SentencePreProcessor extends Serializable {

    /**
     * Pre process a sentence
     * @param sentence the sentence to pre process
     * @return the pre processed sentence
     */
    String preProcess(String sentence);

}
