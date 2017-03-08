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

package org.deeplearning4j.spark.text.functions;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizerfactory.NGramTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.util.Arrays;
import java.util.List;

/**
 * Tokenizer function
 * @author Adam Gibson
 */
@SuppressWarnings("unchecked")
public class TokenizerFunction implements Function<String, List<String>> {
    private String tokenizerFactoryClazz;
    private String tokenizerPreprocessorClazz;
    private transient TokenizerFactory tokenizerFactory;
    private int nGrams = 1;

    public TokenizerFunction(String tokenizer, String tokenizerPreprocessor, int nGrams) {
        this.tokenizerFactoryClazz = tokenizer;
        this.tokenizerPreprocessorClazz = tokenizerPreprocessor;
        this.nGrams = nGrams;
    }

    @Override
    public List<String> call(String v1) throws Exception {
        if (tokenizerFactory == null)
            tokenizerFactory = getTokenizerFactory();
        if (v1.isEmpty())
            return Arrays.asList("");
        return tokenizerFactory.create(v1).getTokens();
    }

    private TokenizerFactory getTokenizerFactory() {
        try {
            TokenPreProcess tokenPreProcessInst = null;
            // token preprocess CAN be undefined
            if (tokenizerPreprocessorClazz != null && !tokenizerPreprocessorClazz.isEmpty()) {
                Class<? extends TokenPreProcess> clazz =
                                (Class<? extends TokenPreProcess>) Class.forName(tokenizerPreprocessorClazz);
                tokenPreProcessInst = clazz.newInstance();
            }

            Class<? extends TokenizerFactory> clazz2 =
                            (Class<? extends TokenizerFactory>) Class.forName(tokenizerFactoryClazz);
            tokenizerFactory = clazz2.newInstance();
            if (tokenPreProcessInst != null)
                tokenizerFactory.setTokenPreProcessor(tokenPreProcessInst);
            if (nGrams > 1) {
                tokenizerFactory = new NGramTokenizerFactory(tokenizerFactory, nGrams, nGrams);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return tokenizerFactory;
    }

}
