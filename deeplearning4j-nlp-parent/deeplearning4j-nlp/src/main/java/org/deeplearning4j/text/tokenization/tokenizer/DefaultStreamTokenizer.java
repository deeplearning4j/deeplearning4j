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


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Tokenizer based on the {@link java.io.StreamTokenizer}
 * @author Adam Gibson
 *
 */
public class DefaultStreamTokenizer implements Tokenizer {

    private StreamTokenizer streamTokenizer;
    private TokenPreProcess tokenPreProcess;
    private List<String> tokens = new ArrayList<>();
    private AtomicInteger position = new AtomicInteger(0);

    protected static final Logger log = LoggerFactory.getLogger(DefaultStreamTokenizer.class);

    public DefaultStreamTokenizer(InputStream is) {
        Reader r = new BufferedReader(new InputStreamReader(is));
        streamTokenizer = new StreamTokenizer(r);

    }

    /**
     * Checks, if underlying stream has any tokens left
     *
     * @return
     */
    private boolean streamHasMoreTokens() {
        if (streamTokenizer.ttype != StreamTokenizer.TT_EOF) {
            try {
                streamTokenizer.nextToken();
            } catch (IOException e1) {
                throw new RuntimeException(e1);
            }
        }
        return streamTokenizer.ttype != StreamTokenizer.TT_EOF && streamTokenizer.ttype != -1;
    }

    /**
     * Checks, if any prebuffered tokens left, otherswise checks underlying stream
     * @return
     */
    @Override
    public boolean hasMoreTokens() {
        log.info("Tokens size: [" + tokens.size() + "], position: [" + position.get() + "]");
        if (!tokens.isEmpty())
            return position.get() < tokens.size();
        else
            return streamHasMoreTokens();
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

    /**
     * This method returns next token from underlying InputStream
     *
     * @return
     */
    private String nextTokenFromStream() {
        StringBuilder sb = new StringBuilder();


        if (streamTokenizer.ttype == StreamTokenizer.TT_WORD) {
            sb.append(streamTokenizer.sval);
        } else if (streamTokenizer.ttype == StreamTokenizer.TT_NUMBER) {
            sb.append(streamTokenizer.nval);
        } else if (streamTokenizer.ttype == StreamTokenizer.TT_EOL) {
            try {
                while (streamTokenizer.ttype == StreamTokenizer.TT_EOL)
                    streamTokenizer.nextToken();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        } else if (streamHasMoreTokens())
            return nextTokenFromStream();


        String ret = sb.toString();

        if (tokenPreProcess != null)
            ret = tokenPreProcess.preProcess(ret);
        return ret;

    }

    /**
     * Returns all tokens as list of Strings
     *
     * @return List of tokens
     */
    @Override
    public List<String> getTokens() {
        //List<String> tokens = new ArrayList<>();
        if (!tokens.isEmpty())
            return tokens;

        log.info("Starting prebuffering...");
        while (streamHasMoreTokens()) {
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
