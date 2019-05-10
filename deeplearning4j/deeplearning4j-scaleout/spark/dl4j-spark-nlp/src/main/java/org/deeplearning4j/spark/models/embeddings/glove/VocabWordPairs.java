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

package org.deeplearning4j.spark.models.embeddings.glove;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.primitives.Triple;

/**
 * Convert string to vocab words
 *
 * @author Adam Gibson
 */
public class VocabWordPairs implements Function<Triple<String, String, Double>, Triple<VocabWord, VocabWord, Double>> {
    private Broadcast<VocabCache<VocabWord>> vocab;

    public VocabWordPairs(Broadcast<VocabCache<VocabWord>> vocab) {
        this.vocab = vocab;
    }

    @Override
    public Triple<VocabWord, VocabWord, Double> call(Triple<String, String, Double> v1) throws Exception {
        return new Triple<>((VocabWord) vocab.getValue().wordFor(v1.getFirst()),
                        (VocabWord) vocab.getValue().wordFor(v1.getSecond()), v1.getThird());
    }
}
