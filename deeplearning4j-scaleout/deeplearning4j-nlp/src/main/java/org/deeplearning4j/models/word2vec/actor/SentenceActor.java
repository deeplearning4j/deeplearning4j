package org.deeplearning4j.models.word2vec.actor;


import akka.actor.ActorRef;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import akka.actor.UntypedActor;


public class SentenceActor extends UntypedActor {

    private Word2Vec vec;
    private ActorRef skipGramActor;
    private static Logger log = LoggerFactory.getLogger(SentenceActor.class);

    public SentenceActor(Word2Vec vec,ActorRef skipGramActor) {
        super();
        this.vec = vec;
        this.skipGramActor = skipGramActor;
    }




    @Override
    public void onReceive(final Object message) throws Exception {
        if(message instanceof SentenceMessage) {

            SentenceMessage m2 = (SentenceMessage) message;
            m2.getChanged().incrementAndGet();
            vec.processSentence(m2.getSentence(),skipGramActor);
            m2.getChanged().decrementAndGet();


        }

        else
            unhandled(message);


    }





}
