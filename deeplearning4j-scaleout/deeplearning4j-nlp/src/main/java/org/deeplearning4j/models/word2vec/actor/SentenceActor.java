package org.deeplearning4j.models.word2vec.actor;


import akka.actor.ActorRef;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import akka.actor.UntypedActor;

import java.util.Collection;


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
            SentenceMessage message2 = (SentenceMessage) message;
            processMessage(message2);

        }

        else if(message instanceof Collection) {
            Collection<SentenceMessage> message2 = (Collection<SentenceMessage>) message;
            for(SentenceMessage message3 : message2) {
                processMessage(message3);
            }
        }

        else
            unhandled(message);


    }

    private void processMessage(SentenceMessage message) {

        SentenceMessage m2 =  message;
        m2.getChanged().incrementAndGet();
        vec.processSentence(m2.getSentence(),skipGramActor);
        m2.getChanged().decrementAndGet();

    }






}
