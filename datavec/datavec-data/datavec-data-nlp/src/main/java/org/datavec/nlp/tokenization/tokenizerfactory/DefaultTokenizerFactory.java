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

package org.datavec.nlp.tokenization.tokenizerfactory;



import org.datavec.nlp.tokenization.tokenizer.DefaultStreamTokenizer;
import org.datavec.nlp.tokenization.tokenizer.DefaultTokenizer;
import org.datavec.nlp.tokenization.tokenizer.TokenPreProcess;
import org.datavec.nlp.tokenization.tokenizer.Tokenizer;

import java.io.InputStream;

/**
 * Default tokenizer based on string tokenizer or stream tokenizer
 * @author Adam Gibson
 */
public class DefaultTokenizerFactory implements TokenizerFactory {

    private TokenPreProcess tokenPreProcess;

    @Override
    public Tokenizer create(String toTokenize) {
        DefaultTokenizer t = new DefaultTokenizer(toTokenize);
        t.setTokenPreProcessor(tokenPreProcess);
        return t;
    }

    @Override
    public Tokenizer create(InputStream toTokenize) {
        Tokenizer t = new DefaultStreamTokenizer(toTokenize);
        t.setTokenPreProcessor(tokenPreProcess);
        return t;
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess preProcessor) {
        this.tokenPreProcess = preProcessor;
    }


}
