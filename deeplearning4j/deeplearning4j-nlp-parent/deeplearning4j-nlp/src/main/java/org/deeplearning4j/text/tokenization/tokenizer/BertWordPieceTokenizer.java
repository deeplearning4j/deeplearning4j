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

package org.deeplearning4j.text.tokenization.tokenizer;

import java.util.*;
import java.util.regex.Pattern;

/**
 * A tokenizer that works with a vocab from a published bert model
 * @author Paul Dubs
 */
public class BertWordPieceTokenizer implements Tokenizer {

    private final Pattern splitPattern = Pattern.compile("(\\p{javaWhitespace}|((?<=\\p{Punct})|(?=\\p{Punct})))+");
    private final List<String> tokens;
    private int cursor;

    public BertWordPieceTokenizer(String tokens, NavigableMap<String, Integer> vocab) {
        this.tokens = tokenize(vocab, tokens);
        this.cursor = 0;
    }

    private TokenPreProcess tokenPreProcess;

    @Override
    public boolean hasMoreTokens() {
        return cursor < tokens.size();
    }

    @Override
    public int countTokens() {
        return tokens.size();
    }

    @Override
    public String nextToken() {
        String base = tokens.get(cursor);
        cursor++;
        if (tokenPreProcess != null)
            base = tokenPreProcess.preProcess(base);
        return base;
    }

    @Override
    public List<String> getTokens() {
        return tokens;
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess tokenPreProcessor) {
        this.tokenPreProcess = tokenPreProcessor;

    }

    private List<String> tokenize(NavigableMap<String, Integer> vocab, String toTokenzie) {
        final List<String> output = new ArrayList<>();

        for (String basicToken : splitPattern.split(toTokenzie)) {
            String candidate = basicToken;

            while(candidate.length() > 0 && !"##".equals(candidate)){
                final Set<Map.Entry<String, Integer>> entries = vocab.headMap(candidate, true).descendingMap().entrySet();
                String longestSubstring = null;
                for (Map.Entry<String, Integer> entry : entries) {
                    if(candidate.startsWith(entry.getKey())){
                        longestSubstring = entry.getKey();
                        break;
                    }
                }
                output.add(longestSubstring);
                candidate = "##"+candidate.substring(longestSubstring.length());
            }
        }

        return output;
    }

}
