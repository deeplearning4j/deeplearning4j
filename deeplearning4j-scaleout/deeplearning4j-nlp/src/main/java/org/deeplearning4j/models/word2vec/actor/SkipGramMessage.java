/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.models.word2vec.actor;

import org.deeplearning4j.models.word2vec.VocabWord;

import java.io.Serializable;
import java.util.List;

/**
 * Created by agibsonccc on 9/16/14.
 */
public class SkipGramMessage implements Serializable {

    private int i;
    private int b;
    private List<VocabWord> sentence;

    public SkipGramMessage(int i, int b, List<VocabWord> sentence) {
        this.i = i;
        this.b = b;
        this.sentence = sentence;
    }

    public int getB() {
        return b;
    }

    public void setB(int b) {
        this.b = b;
    }

    public List<VocabWord> getSentence() {
        return sentence;
    }

    public void setSentence(List<VocabWord> sentence) {
        this.sentence = sentence;
    }

    public int getI() {

        return i;
    }

    public void setI(int i) {
        this.i = i;
    }
}
