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

package org.deeplearning4j.text.tokenization.tokenizerfactory;

import org.deeplearning4j.text.tokenization.tokenizer.KoreanTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;

import java.io.InputStream;

/**
 * Created by kepricon on 16. 10. 20.
 */
public class KoreanTokenizerFactory implements TokenizerFactory {

    private TokenPreProcess preProcess;

    public KoreanTokenizerFactory() {}

    @Override
    public Tokenizer create(String toTokenize) {
        KoreanTokenizer t = new KoreanTokenizer(toTokenize);
        t.setTokenPreProcessor(preProcess);
        return t;
    }

    @Override
    public Tokenizer create(InputStream inputStream) {
        throw new UnsupportedOperationException("Not supported");
        //        return null;
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess tokenPreProcess) {
        this.preProcess = tokenPreProcess;
    }

    @Override
    public TokenPreProcess getTokenPreProcessor() {
        return this.preProcess;
    }
}
