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

package org.deeplearning4j.text.tokenization.tokenizerFactory;

import org.deeplearning4j.text.tokenization.tokenizer.ChineseTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.InputStream;

/**
 * @date: June 2,2017
 * @author: wangfeng
 * @Description:
 */

public class ChineseTokenizerFactory implements TokenizerFactory {

    private TokenPreProcess tokenPreProcess;

    @Override
    public Tokenizer create(String toTokenize) {
        Tokenizer tokenizer = new ChineseTokenizer(toTokenize);
        tokenizer.setTokenPreProcessor(tokenPreProcess);
        return tokenizer;
    }

    @Override
    public Tokenizer create(InputStream toTokenize) {
        throw new UnsupportedOperationException();
        /*  Tokenizer t =  new ChineseStreamTokenizer(toTokenize);
        t.setTokenPreProcessor(tokenPreProcess);
        return t;*/
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess tokenPreProcess) {
        this.tokenPreProcess = tokenPreProcess;
    }

    @Override
    public TokenPreProcess getTokenPreProcessor() {
        return tokenPreProcess;
    }
}
