/*
 *  * Copyright 2017 Skymind, Inc.
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

package org.datavec.api.transform.nlp.impl;

import org.datavec.api.transform.nlp.Tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.regex.Pattern;

/**
 * Default tokenizer
 * @author Adam Gibson
 */
public class DefaultTokenizer implements Tokenizer {
    private static final Pattern punctPattern = Pattern.compile("[\\d\\.:,\"\'\\(\\)\\[\\]|/?!;]+");

    public DefaultTokenizer(String tokens) {
        tokenizer = new StringTokenizer(tokens);
    }

    private StringTokenizer tokenizer;

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
        return tokenizer.nextToken();
    }

    @Override
    public List<String> getTokens() {
        List<String> tokens = new ArrayList<>();
        while (hasMoreTokens()) {
            String token = nextToken();
            tokens.add(punctPattern.matcher(token).replaceAll(""));
        }
        return tokens;
    }
}
