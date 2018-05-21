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

import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * @author sonali
 */
public class NGramTokenizer implements Tokenizer {
    private List<String> tokens;
    private List<String> originalTokens;
    private int index;
    private TokenPreProcess preProcess;
    private Tokenizer tokenizer;

    public NGramTokenizer(Tokenizer tokenizer, Integer minN, Integer maxN) {
        this.tokens = new ArrayList<>();
        while (tokenizer.hasMoreTokens()) {
            String nextToken = tokenizer.nextToken();
            this.tokens.add(nextToken);
        }
        if (maxN != 1) {
            this.originalTokens = this.tokens;
            this.tokens = new ArrayList<>();
            Integer nOriginalTokens = this.originalTokens.size();
            Integer min = Math.min(maxN + 1, nOriginalTokens + 1);
            for (int i = minN; i < min; i++) {
                for (int j = 0; j < nOriginalTokens - i + 1; j++) {
                    List<String> originalTokensSlice = this.originalTokens.subList(j, j + i);
                    this.tokens.add(StringUtils.join(originalTokensSlice, " "));
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
        while (hasMoreTokens()) {
            tokens.add(nextToken());
        }
        return tokens;
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess tokenPreProcessor) {
        this.preProcess = tokenPreProcessor;
    }
}
