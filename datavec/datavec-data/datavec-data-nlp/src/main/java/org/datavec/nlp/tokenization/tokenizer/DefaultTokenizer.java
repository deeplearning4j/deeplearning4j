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

package org.datavec.nlp.tokenization.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

/**
 * Default tokenizer
 * @author Adam Gibson
 */
public class DefaultTokenizer implements Tokenizer {

    public DefaultTokenizer(String tokens) {
        tokenizer = new StringTokenizer(tokens);
    }

    private StringTokenizer tokenizer;
    private TokenPreProcess tokenPreProcess;

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
        String base = tokenizer.nextToken();
        if (tokenPreProcess != null)
            base = tokenPreProcess.preProcess(base);
        return base;
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
