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

package org.deeplearning4j.models.word2vec.wordstore;

import lombok.Data;
import lombok.NonNull;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;

import java.io.IOException;
import java.io.Serializable;

/**
 * Simplified version of VocabWord.
 * Used only for w2v vocab building routines
 *
 * @author raver119@gmail.com
 */

@Data
public class VocabularyWord implements Serializable {
    @NonNull
    private String word;
    private int count = 1;

    // these fields are used for vocab serialization/deserialization only. Usual runtime value is null.
    private double[] syn0;
    private double[] syn1;
    private double[] syn1Neg;
    private double[] historicalGradient;

    private static ObjectMapper mapper;
    private static final Object lock = new Object();

    // There's no reasons to save HuffmanNode data, it will be recalculated after deserialization
    private transient HuffmanNode huffmanNode;

    // empty constructor is required for proper deserialization
    public VocabularyWord() {

    }

    public VocabularyWord(@NonNull String word) {
        this.word = word;
    }

    /*
        since scavenging mechanics are targeting low-freq words, byte values is definitely enough.
        please note, array is initialized outside of this scope, since it's only holder, no logic involved inside this class
      */
    private transient byte[] frequencyShift;
    private transient byte retentionStep;

    // special mark means that this word should NOT be affected by minWordFrequency setting
    private boolean special = false;

    public void incrementCount() {
        this.count++;
    }

    public void incrementRetentionStep() {
        this.retentionStep += 1;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        VocabularyWord word1 = (VocabularyWord) o;

        return word != null ? word.equals(word1.word) : word1.word == null;

    }

    @Override
    public int hashCode() {
        return word != null ? word.hashCode() : 0;
    }

    private static ObjectMapper mapper() {
        if (mapper == null) {
            synchronized (lock) {
                if (mapper == null) {
                    mapper = new ObjectMapper();
                    mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
                    mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
                    mapper.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
                    return mapper;
                }
            }
        }
        return mapper;
    }

    public String toJson() {
        ObjectMapper mapper = mapper();
        try {
            /*
                we need JSON as single line to save it at first line of the CSV model file
            */
            return mapper.writeValueAsString(this);
        } catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    public static VocabularyWord fromJson(String json) {
        ObjectMapper mapper = mapper();
        try {
            VocabularyWord ret = mapper.readValue(json, VocabularyWord.class);
            return ret;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
