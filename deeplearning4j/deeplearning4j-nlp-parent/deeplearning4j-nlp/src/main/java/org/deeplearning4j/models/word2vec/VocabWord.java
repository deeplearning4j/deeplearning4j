/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.models.word2vec;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.Serializable;


/**
 * Intermediate layers of the neural network
 *
 * @author Adam Gibson
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class", defaultImpl =  VocabWord.class)
@JsonAutoDetect(fieldVisibility = JsonAutoDetect.Visibility.ANY, getterVisibility = JsonAutoDetect.Visibility.NONE,
        setterVisibility = JsonAutoDetect.Visibility.NONE)
public class VocabWord extends SequenceElement implements Serializable {

    private static final long serialVersionUID = 2223750736522624256L;

    //for my sanity
    private String word;

    /*
        Used for Joint/Distributed vocabs mechanics
     */
    @Getter
    @Setter
    protected long vocabId;
    @Getter
    @Setter
    protected long affinityId;

    public static VocabWord none() {
        return new VocabWord(0, "none");
    }

    /**
     *
     * @param wordFrequency count of the word
    
     */
    public VocabWord(double wordFrequency, @NonNull String word) {
        if (word.isEmpty())
            throw new IllegalArgumentException("Word must not be null or empty");
        this.word = word;
        this.elementFrequency.set(wordFrequency);
        this.storageId = SequenceElement.getLongHash(word);
    }

    public VocabWord(double wordFrequency, @NonNull String word, long storageId) {
        this(wordFrequency, word);
        this.storageId = storageId;
    }


    public VocabWord() {}


    public String getLabel() {
        return this.word;
    }


    public String getWord() {
        return word;
    }

    public void setWord(String word) {
        this.word = word;
    }


    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof VocabWord))
            return false;
        final VocabWord vocabWord = (VocabWord) o;
        if (this.word == null)
            return vocabWord.word == null;

        return this.word.equals(vocabWord.getWord());
    }
    

    @Override
    public int hashCode() {
        final int result = this.word == null ? 0 : this.word.hashCode(); //this.elementFrequency.hashCode();
        return result;
    }

    @Override
    public String toString() {
        return "VocabWord{" +
                "word='" + word + '\'' +
                ", elementFrequency=" + elementFrequency +
                ", index=" + index +
                ", codes=" + codes +
                ", points=" + points +
                ", codeLength=" + codeLength +
                '}';
    }

    @Override
    public String toJSON() {
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


}

