package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.word2vec.VocabWord;

import java.io.Serializable;
import java.util.List;

/**
 * Created by agibsoncccc on 4/20/15.
 */
public class Word2VecFuncCall implements Serializable {
    private Broadcast<Word2VecParam> param;
    private Long wordsSeen;
    private List<VocabWord> sentence;

    public Word2VecFuncCall(Broadcast<Word2VecParam> param, Long wordsSeen, List<VocabWord> sentence) {
        this.param = param;
        this.wordsSeen = wordsSeen;
        this.sentence = sentence;
    }

    public Broadcast<Word2VecParam>  getParam() {
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
