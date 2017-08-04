/*-
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.text.tokenization.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

/**
 * Default tokenizer
 * @author Adam Gibson
 */
public class DefaultTokenizer implements Tokenizer {

    public DefaultTokenizer(String tokens) {
        tokenizer = new StringTokenizer(tokens);
    }

    private StringTokenizer tokenizer;
    private TokenPreProcess tokenPreProcess;

    @Override
    public boolean hasMoreTokens() {
        return tokenizer.hasMoreTokens();
    }

    @Override
    public int countTokens() {
        return tokenizer.countTokens();
    }

    @Override
    public String nextToken() {
        String base = tokenizer.nextToken();
        if (tokenPreProcess != null)
            base = tokenPreProcess.preProcess(base);
        return base;
    }

    @Override
    public List<String> getTokens() {
        List<String> tokens = new ArrayList<>();
        while (hasMoreTokens()) {
            tokens.add(nextToken());
        }
        return tokens;
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess tokenPreProcessor) {
        this.tokenPreProcess = tokenPreProcessor;

    }



}
