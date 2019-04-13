/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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


import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A tokenizer that works with a vocab from a published bert model and tokenizes a token at a time from a stream
 * @author Paul Dubs
 */
@Slf4j
public class BertWordPieceStreamTokenizer implements Tokenizer {
    private final NavigableMap<String, Integer> vocab;
    private final Reader reader;
    private final boolean lowerCaseOnly;
    private boolean more = true;
    private String buffer = "";
    private int longestToken = 0;
    private String prevRest = null;
    private boolean noSplit = false;

    private TokenPreProcess tokenPreProcess;
    private List<String> tokens = new ArrayList<>();
    private AtomicInteger position = new AtomicInteger(0);

    public BertWordPieceStreamTokenizer(InputStream is, NavigableMap<String, Integer> vocab, boolean lowerCaseOnly) {
        this.lowerCaseOnly = lowerCaseOnly;
        if(vocab.comparator() == null || vocab.comparator().compare("a", "b") < 0){
            throw new IllegalArgumentException("Vocab must use reverse sort order!");
        }

        this.reader = new BufferedReader(new InputStreamReader(is));
        this.vocab = vocab;
        for (String token : vocab.keySet()) {
            if(token.length() > longestToken){
                longestToken = token.length();
            }
        }
    }

    /**
     * Checks, if any prebuffered tokens left, otherswise checks underlying stream
     * @return
     */
    @Override
    public boolean hasMoreTokens() {
        return more || buffer.length() > 0 || prevRest != null;
    }

    private void readMore(){
        final StringBuilder builder = new StringBuilder(longestToken);
        while(more && builder.length() < longestToken){
            try{
                final int codePoint = reader.read();
                if(codePoint >= 0){
                    builder.appendCodePoint(codePoint);
                }else{
                    more = false;
                }
            } catch (IOException e) {
                more = false;
                log.error("Unexpected exception while reading input stream", e);
            }
        }

        String input = builder.toString();
        if(lowerCaseOnly){
            input = input.toLowerCase();
        }
        if(noSplit){
            final String[] parts = BertWordPieceTokenizer.splitPattern.split(input, 2);
            prevRest = (prevRest == null ? "" : prevRest) + parts[0];
            if(parts.length > 1){
                noSplit = false;
                buffer += parts[1];
            }
        }else{
            buffer += input;
        }
    }

    /**
     * Returns number of tokens
     * PLEASE NOTE: this method effectively preloads all tokens. So use it with caution, since on large streams it will consume big amount of memory
     *
     * @return
     */
    @Override
    public int countTokens() {
        return getTokens().size();
    }


    /**
     * This method returns next token from prebuffered list of tokens or underlying InputStream
     *
     * @return next token as String
     */
    @Override
    public String nextToken() {
        if (!tokens.isEmpty() && position.get() < tokens.size())
            return tokens.get(position.getAndIncrement());

        return nextTokenFromStream();
    }

    private String nextTokenFromStream(){
        if(noSplit && more) readMore();
        String basicToken = prevRest;
        if(basicToken == null || basicToken.length() == 0){
            if(buffer.length() < longestToken && more) readMore();
            final String[] parts = BertWordPieceTokenizer.splitPattern.split(buffer, 2);
            basicToken = parts[0];
            if(parts.length > 1){
                buffer = parts[1];
                noSplit = false;
            }else{
                buffer = "";
                noSplit = true;
            }
        }

        String output = BertWordPieceTokenizer.findLongestSubstring(vocab, basicToken);
        String tokenRest = basicToken.substring(output.length());
        if(basicToken.length() > output.length()){
            tokenRest = "##"+tokenRest;
        }

        if("##".equals(tokenRest) || tokenRest.length() == 0) tokenRest = null;

        prevRest = tokenRest;

        if (tokenPreProcess != null)
            output = tokenPreProcess.preProcess(output);

        return output;
    }

    /**
     * Returns all tokens as list of Strings
     *
     * @return List of tokens
     */
    @Override
    public List<String> getTokens() {
        if (!tokens.isEmpty())
            return tokens;

        log.info("Starting prebuffering...");
        while (hasMoreTokens()) {
            tokens.add(nextTokenFromStream());
        }
        log.info("Tokens prefetch finished. Tokens size: [" + tokens.size() + "]");
        return tokens;
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess tokenPreProcessor) {
        this.tokenPreProcess = tokenPreProcessor;
    }

}
