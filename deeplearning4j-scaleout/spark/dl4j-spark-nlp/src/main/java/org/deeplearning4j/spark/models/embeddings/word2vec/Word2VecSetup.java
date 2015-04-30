package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.word2vec.VocabWord;
import scala.Tuple2;

import java.util.List;

/**
 * Created by agibsoncccc on 4/20/15.
 */
public class Word2VecSetup implements Function<Tuple2<List<VocabWord>, Long>, Word2VecFuncCall> {
    private Broadcast<Word2VecParam> param;

    public Word2VecSetup(Broadcast<Word2VecParam> param) {
        this.param = param;
    }

    @Override
    public Word2VecFuncCall call(Tuple2<List<VocabWord>, Long> listLongTuple2) throws Exception {
        return new Word2VecFuncCall(param,listLongTuple2._2(),listLongTuple2._1());
    }
}
