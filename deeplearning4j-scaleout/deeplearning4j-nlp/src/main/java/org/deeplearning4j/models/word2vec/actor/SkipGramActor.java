package org.deeplearning4j.models.word2vec.actor;

import akka.actor.UntypedActor;
import org.deeplearning4j.models.word2vec.Word2Vec;

/**
 * Created by agibsonccc on 9/16/14.
 */
public class SkipGramActor extends UntypedActor {
    private Word2Vec vec;

    public SkipGramActor(Word2Vec vec) {
        this.vec = vec;
    }

    /**
     * To be implemented by concrete UntypedActor, this defines the behavior of the
     * UntypedActor.
     *
     * @param message
     */
    @Override
    public void onReceive(Object message) throws Exception {
        if(message instanceof SkipGramMessage) {
            SkipGramMessage m = (SkipGramMessage) message;
            vec.skipGram(m.getI(),m.getSentence(),m.getB());
        }
        else
            unhandled(message);
    }
}
