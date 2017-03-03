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

import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.word2vec.VocabWord;
import scala.Tuple2;

import java.util.List;

/**
 * Set up word2vec to run an iteration
 *
 * @author Adam Gibson
 */
@Deprecated
public class Word2VecSetup implements Function<Tuple2<List<VocabWord>, Long>, Word2VecFuncCall> {
    private Broadcast<Word2VecParam> param;

    public Word2VecSetup(Broadcast<Word2VecParam> param) {
        this.param = param;
    }

    @Override
    public Word2VecFuncCall call(Tuple2<List<VocabWord>, Long> listLongTuple2) throws Exception {
        return new Word2VecFuncCall(param, listLongTuple2._2(), listLongTuple2._1());
    }
}
