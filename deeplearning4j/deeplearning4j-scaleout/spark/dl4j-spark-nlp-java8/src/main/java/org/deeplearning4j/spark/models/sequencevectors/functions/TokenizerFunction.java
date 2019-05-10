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

package org.deeplearning4j.spark.models.sequencevectors.functions;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.word2vec.VocabWord;

import java.util.List;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class TokenizerFunction extends BaseTokenizerFunction implements Function<String, Sequence<VocabWord>> {

    public TokenizerFunction(@NonNull Broadcast<VectorsConfiguration> configurationBroadcast) {
        super(configurationBroadcast);
    }


    @Override
    public Sequence<VocabWord> call(String s) throws Exception {
        if (tokenizerFactory == null)
            instantiateTokenizerFactory();

        List<String> tokens = tokenizerFactory.create(s).getTokens();
        Sequence<VocabWord> seq = new Sequence<>();
        for (String token : tokens) {
            if (token == null || token.isEmpty())
                continue;

            seq.addElement(new VocabWord(1.0, token));
        }

        return seq;
    }
}
