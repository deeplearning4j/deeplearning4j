/*-
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

package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.word2vec.VocabWord;

import java.io.Serializable;
import java.util.List;

/**
 * Map operation for word2vec
 *
 * @author dAdam Gibson
 */
@Deprecated
public class Word2VecFuncCall implements Serializable {
    private Broadcast<Word2VecParam> param;
    private Long wordsSeen;
    private List<VocabWord> sentence;

    public Word2VecFuncCall(Broadcast<Word2VecParam> param, Long wordsSeen, List<VocabWord> sentence) {
        this.param = param;
        this.wordsSeen = wordsSeen;
        this.sentence = sentence;
    }

    public Broadcast<Word2VecParam> getParam() {
        return param;
    }

    public void setParam(Broadcast<Word2VecParam> param) {
        this.param = param;
    }

    public Long getWordsSeen() {
        return wordsSeen;
    }

    public void setWordsSeen(Long wordsSeen) {
        this.wordsSeen = wordsSeen;
    }

    public List<VocabWord> getSentence() {
        return sentence;
    }

    public void setSentence(List<VocabWord> sentence) {
        this.sentence = sentence;
    }
}
