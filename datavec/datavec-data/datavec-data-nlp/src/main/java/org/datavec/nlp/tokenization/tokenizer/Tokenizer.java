/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.nlp.tokenization.tokenizer;

import java.util.List;

/**
 * A representation of a tokenizer.
 * Different applications may require 
 * different kind of tokenization (say rules based vs more formal NLP approaches)
 * @author Adam Gibson
 *
 */
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
