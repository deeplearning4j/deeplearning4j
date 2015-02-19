/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.spark.text;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.util.List;

/**
 * Tokenizer function
 * @author Adam Gibson
 */
public class TokenizerFunction implements Function<String,Pair<List<String>,Long>> {
    private Class<? extends TokenizerFactory> tokenizerFactoryClazz;
    private TokenizerFactory tokenizerFactory;

    public TokenizerFunction(String clazz) {
        try {
            tokenizerFactoryClazz = (Class<? extends TokenizerFactory>) Class.forName(clazz);
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

    }

    public TokenizerFunction() {
        this(DefaultTokenizerFactory.class.getName());
    }

    @Override
    public Pair<List<String>,Long> call(String v1) throws Exception {
        if(tokenizerFactory == null)
            tokenizerFactory = getTokenizerFactory();
        List<String> tokens = tokenizerFactory.create(v1).getTokens();
        return new Pair<>(tokens, (long) tokens.size());
    }
    private TokenizerFactory getTokenizerFactory() {
        try {
            tokenizerFactory = tokenizerFactoryClazz.newInstance();
        } catch (InstantiationException | IllegalAccessException e) {
            e.printStackTrace();
        }
      return tokenizerFactory;
    }

}
