/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

package org.deeplearning4j.spark.text.functions;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.common.config.DL4JClassLoading;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizerfactory.NGramTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.util.Collections;
import java.util.List;

/**
 * Tokenizer function
 * @author Adam Gibson
 */
@SuppressWarnings("unchecked")
@Slf4j
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
    public List<String> call(String str) {
        if (tokenizerFactory == null) {
            tokenizerFactory = getTokenizerFactory();
        }

        if (str.isEmpty()) {
            return Collections.singletonList("");
        }

        return tokenizerFactory.create(str).getTokens();
    }

    private TokenizerFactory getTokenizerFactory() {
        TokenPreProcess tokenPreProcessInst = null;

        if (StringUtils.isNotEmpty(tokenizerPreprocessorClazz)) {
            tokenPreProcessInst = DL4JClassLoading.createNewInstance(tokenizerPreprocessorClazz);
        }

        tokenizerFactory = DL4JClassLoading.createNewInstance(tokenizerFactoryClazz);

        if (tokenPreProcessInst != null)
            tokenizerFactory.setTokenPreProcessor(tokenPreProcessInst);
        if (nGrams > 1) {
            tokenizerFactory = new NGramTokenizerFactory(tokenizerFactory, nGrams, nGrams);
        }

        return tokenizerFactory;
    }

}
