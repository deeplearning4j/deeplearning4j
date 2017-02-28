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


import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Tokenizer based on the {@link java.io.StreamTokenizer}
 * @author Adam Gibson
 *
 */
public class DefaultStreamTokenizer implements Tokenizer {

    private StreamTokenizer streamTokenizer;
    private TokenPreProcess tokenPreProcess;


    public DefaultStreamTokenizer(InputStream is) {
        Reader r = new BufferedReader(new InputStreamReader(is));
        streamTokenizer = new StreamTokenizer(r);

    }

    @Override
    public boolean hasMoreTokens() {
        if (streamTokenizer.ttype != StreamTokenizer.TT_EOF) {
            try {
                streamTokenizer.nextToken();
            } catch (IOException e1) {
                throw new RuntimeException(e1);
            }
        }
        return streamTokenizer.ttype != StreamTokenizer.TT_EOF && streamTokenizer.ttype != -1;
    }

    @Override
    public int countTokens() {
        return getTokens().size();
    }

    @Override
    public String nextToken() {
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
        }

        else if (hasMoreTokens())
            return nextToken();


        String ret = sb.toString();

        if (tokenPreProcess != null)
            ret = tokenPreProcess.preProcess(ret);
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
        this.tokenPreProcess = tokenPreProcessor;
    }

}
