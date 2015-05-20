/*
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

import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;

/**
 * @author sonali
 */
public class NGramTokenizer implements Tokenizer {
    private List<String> tokens;
    private List<String> originalTokens;
    private int index;
    private TokenPreProcess preProcess;
    private StringTokenizer tokenizer;

    public NGramTokenizer(String tokens, Integer minN, Integer maxN) {
        tokenizer = new StringTokenizer(tokens);
        while (tokenizer.hasMoreTokens()) {
            this.tokens.add(tokenizer.nextToken());
        }
        if (maxN != 1) {
            this.originalTokens = this.tokens;
            this.tokens = new ArrayList<String>();
            Integer nOriginalTokens = originalTokens.size();
            Integer min = Math.min(maxN + 1, nOriginalTokens + 1);
            for (int i = minN; i < min; i++) {
                for (int j = 0; j < nOriginalTokens - i + 1; j++) {
                    String[] originalTokensSlice = Arrays.copyOfRange(originalTokens, j, j + i);
                    this.tokens.add(StringUtils.join(" ", originalTokensSlice));
                }
            }
        }
    }

    @Override
    public boolean hasMoreTokens() {
        return index < tokens.size();
    }

    @Override
    public int countTokens() {
        return tokens.size();
    }

    @Override
    public String nextToken() {
        String ret = tokens.get(index);
        index++;
        return ret;
    }

    @Override
    public List<String> getTokens() {
        List<String> tokens = new ArrayList<>();
        while(hasMoreTokens()) {
            tokens.add(nextToken());
        }
        return tokens;
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess tokenPreProcessor) {
        this.preProcess = tokenPreProcessor;
    }
}
