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

package org.deeplearning4j.text.tokenization.tokenizer;

import java.util.List;

public interface Tokenizer {

    /**
     * An iterator for tracking whether
     * more tokens are left in the iterator not
     * @return whether there is anymore tokens
     * to iterate over
     */
    boolean hasMoreTokens();

    /**
     * The number of tokens in the tokenizer
     * @return the number of tokens
     */
    int countTokens();

    /**
     * The next token (word usually) in the string
     * @return the next token in the string if any
     */
    String nextToken();

    /**
     * Returns a list of all the tokens
     * @return a list of all the tokens
     */
    List<String> getTokens();

    /**
     * Set the token pre process
     * @param tokenPreProcessor the token pre processor to set
     */
    void setTokenPreProcessor(TokenPreProcess tokenPreProcessor);



}
