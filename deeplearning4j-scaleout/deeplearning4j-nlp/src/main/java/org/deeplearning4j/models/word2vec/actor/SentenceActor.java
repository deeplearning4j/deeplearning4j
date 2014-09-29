package org.deeplearning4j.models.word2vec.actor;

import akka.actor.UntypedActor;
import org.deeplearning4j.models.word2vec.Word2Vec;

/**
 * Created by agibsonccc on 9/28/14.
 */
public class SentenceActor extends UntypedActor {

    private Word2Vec vec;

    public SentenceActor(Word2Vec vec) {
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
        if(message instanceof String) {
            vec.trainSentence(message.toString());
        }

        else if(message instanceof SentenceMessage) {
            ((SentenceMessage) message).getChanged().incrementAndGet();
            vec.trainSentence(((SentenceMessage) message).getSentence());
            ((SentenceMessage) message).getChanged().incrementAndGet();
        }
    }
}
