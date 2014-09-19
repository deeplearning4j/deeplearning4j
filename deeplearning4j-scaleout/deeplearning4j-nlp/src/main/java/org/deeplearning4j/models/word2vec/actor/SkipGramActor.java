package org.deeplearning4j.models.word2vec.actor;

import akka.actor.UntypedActor;
import akka.dispatch.Futures;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.util.concurrent.Callable;

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
            final SkipGramMessage m = (SkipGramMessage) message;
            Futures.future(new Callable<Void>() {
                /**
                 * Computes a result, or throws an exception if unable to do so.
                 *
                 * @return computed result
                 * @throws Exception if unable to compute a result
                 */
                @Override
                public Void call() throws Exception {
                    vec.skipGram(m.getI(), m.getSentence(), m.getB());
                    return null;
                }
            },context().dispatcher());

        }
        else
            unhandled(message);
    }
}
