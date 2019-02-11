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


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.NavigableMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;

/**
 * A tokenizer that works with a vocab from a published bert model and tokenizes a token at a time from a stream
 * @author Paul Dubs
 */
public class BertWordPieceStreamTokenizer implements Tokenizer {

    private final Pattern splitPattern = Pattern.compile("(\\p{javaWhitespace}|((?<=\\p{Punct})|(?=\\p{Punct})))+");
    private final NavigableMap<String, Integer> vocab;
    private final Reader reader;
    private boolean more = true;
    private String buffer = "";
    private int longestToken = 0;
    private String prevRest = null;
    private boolean noSplit = false;

    private TokenPreProcess tokenPreProcess;
    private List<String> tokens = new ArrayList<>();
    private AtomicInteger position = new AtomicInteger(0);

    protected static final Logger log = LoggerFactory.getLogger(BertWordPieceStreamTokenizer.class);

    public BertWordPieceStreamTokenizer(InputStream is, NavigableMap<String, Integer> vocab) {
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

        if(noSplit){
            final String[] parts = splitPattern.split(builder.toString(), 2);
            prevRest = (prevRest == null ? "" : prevRest) + parts[0];
            if(parts.length > 1){
                noSplit = false;
                buffer += parts[1];
            }
        }else{
            buffer += builder.toString();
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
            final String[] parts = splitPattern.split(buffer, 2);
            basicToken = parts[0];
            if(parts.length > 1){
                buffer = parts[1];
                noSplit = false;
            }else{
                buffer = "";
                noSplit = true;
            }
        }

        String longestSubstring = vocab.floorKey(basicToken);
        int subStringLength = Math.min(basicToken.length(), longestSubstring.length());
        while(!basicToken.startsWith(longestSubstring)){
            subStringLength--;
            longestSubstring = vocab.floorKey(basicToken.substring(0, subStringLength));
        }

        String output = longestSubstring;
        String tokenRest = basicToken.substring(output.length());
        if(basicToken.length() > output.length()){
            tokenRest = "##"+tokenRest;
        }

        if(tokenRest.equals("##") || tokenRest.length() == 0) tokenRest = null;

        prevRest = tokenRest;

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
