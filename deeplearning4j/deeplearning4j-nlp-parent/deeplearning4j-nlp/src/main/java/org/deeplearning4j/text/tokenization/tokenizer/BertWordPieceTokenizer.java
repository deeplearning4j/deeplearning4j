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

package org.deeplearning4j.text.tokenization.tokenizer;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.BertWordPiecePreProcessor;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;

/**
 * A tokenizer that works with a vocab from a published bert model
 * @author Paul Dubs
 */
@Slf4j
public class BertWordPieceTokenizer implements Tokenizer {
    public static final Pattern splitPattern = Pattern.compile("\\p{javaWhitespace}+|((?<=\\p{Punct})+|(?=\\p{Punct}+))");

    private final List<String> tokens;
    private final TokenPreProcess preTokenizePreProcessor;
    private TokenPreProcess tokenPreProcess;
    private final AtomicInteger cursor = new AtomicInteger(0);

    public BertWordPieceTokenizer(String tokens, NavigableMap<String, Integer> vocab, TokenPreProcess preTokenizePreProcessor,
                                  TokenPreProcess tokenPreProcess) {
        if(vocab.comparator() == null || vocab.comparator().compare("a", "b") < 0){
            throw new IllegalArgumentException("Vocab must use reverse sort order!");
        }
        this.preTokenizePreProcessor = preTokenizePreProcessor;
        this.tokenPreProcess = tokenPreProcess;

        this.tokens = tokenize(vocab, tokens);
    }


    @Override
    public boolean hasMoreTokens() {
        return cursor.get() < tokens.size();
    }

    @Override
    public int countTokens() {
        return tokens.size();
    }

    @Override
    public String nextToken() {
        String base = tokens.get(cursor.getAndIncrement());
        if (tokenPreProcess != null)
            base = tokenPreProcess.preProcess(base);
        return base;
    }

    @Override
    public List<String> getTokens() {
        if (tokenPreProcess != null){
            final List<String> result = new ArrayList<>(tokens.size());
            for (String token : tokens) {
                result.add(tokenPreProcess.preProcess(token));
            }
            return result;
        }else {
            return tokens;
        }
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess tokenPreProcessor) {
        this.tokenPreProcess = tokenPreProcessor;

    }

    private List<String> tokenize(NavigableMap<String, Integer> vocab, String toTokenize) {
        final List<String> output = new ArrayList<>();

        String fullString = toTokenize;
        if(preTokenizePreProcessor != null){
            fullString = preTokenizePreProcessor.preProcess(toTokenize);
        }

        for (String basicToken : splitPattern.split(fullString)) {
            String candidate = basicToken;
            int count = 0;
            while(candidate.length() > 0 && !"##".equals(candidate)){
                String longestSubstring = findLongestSubstring(vocab, candidate);
                output.add(longestSubstring);
                candidate = "##"+candidate.substring(longestSubstring.length());
                if(count++ > basicToken.length()){
                    //Can't take more steps to tokenize than the length of the token
                    throw new IllegalStateException("Invalid token encountered: \"" + basicToken + "\" likely contains characters that are not " +
                            "present in the vocabulary. Invalid tokens may be cleaned in a preprocessing step using a TokenPreProcessor." +
                            " preTokenizePreProcessor=" + preTokenizePreProcessor + ", tokenPreProcess=" + tokenPreProcess);
                }
            }
        }

        return output;
    }

    protected String findLongestSubstring(NavigableMap<String, Integer> vocab, String candidate) {
        NavigableMap<String, Integer> tailMap = vocab.tailMap(candidate, true);
        checkIfEmpty(tailMap, candidate);

        String longestSubstring = tailMap.firstKey();
        int subStringLength = Math.min(candidate.length(), longestSubstring.length());
        while(!candidate.startsWith(longestSubstring)){
            subStringLength--;
            tailMap = tailMap.tailMap(candidate.substring(0, subStringLength), true);
            checkIfEmpty(tailMap, candidate);
            longestSubstring = tailMap.firstKey();
        }
        return longestSubstring;
    }

    protected void checkIfEmpty(Map<String,Integer> m, String candidate){
        if(m.isEmpty()){
            throw new IllegalStateException("Invalid token/character encountered: \"" + candidate + "\" likely contains characters that are not " +
                    "present in the vocabulary. Invalid tokens may be cleaned in a preprocessing step using a TokenPreProcessor." +
                    " preTokenizePreProcessor=" + preTokenizePreProcessor + ", tokenPreProcess=" + tokenPreProcess);
        }
    }

}
