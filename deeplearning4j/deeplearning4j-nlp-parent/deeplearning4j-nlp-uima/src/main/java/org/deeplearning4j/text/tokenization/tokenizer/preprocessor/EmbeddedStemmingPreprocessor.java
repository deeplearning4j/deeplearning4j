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

package org.deeplearning4j.text.tokenization.tokenizer.preprocessor;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.tartarus.snowball.ext.PorterStemmer;

/**
 * This tokenizer preprocessor uses given preprocessor + does english Porter stemming on tokens on top of it
 *
 *
 * @author raver119@gmail.com
 */
@NoArgsConstructor
public class EmbeddedStemmingPreprocessor implements TokenPreProcess {
    private TokenPreProcess preProcessor;

    public EmbeddedStemmingPreprocessor(@NonNull TokenPreProcess preProcess) {
        this.preProcessor = preProcess;
    }

    @Override
    public String preProcess(String token) {
        String prep = preProcessor == null ? token : preProcessor.preProcess(token);
        PorterStemmer stemmer = new PorterStemmer();
        stemmer.setCurrent(prep);
        stemmer.stem();

        return stemmer.getCurrent();
    }
}
