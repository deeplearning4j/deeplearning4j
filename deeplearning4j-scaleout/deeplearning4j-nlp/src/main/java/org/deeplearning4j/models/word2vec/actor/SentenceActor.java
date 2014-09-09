package org.deeplearning4j.models.word2vec.actor;

import java.util.concurrent.Callable;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import akka.actor.UntypedActor;
import akka.dispatch.Futures;
import akka.dispatch.OnComplete;
import akka.japi.Creator;

public class SentenceActor extends UntypedActor {

	private Word2Vec vec;
	private static Logger log = LoggerFactory.getLogger(SentenceActor.class);
	
	public SentenceActor(Word2Vec vec) {
		super();
		this.vec = vec;
	}




	@Override
	public void onReceive(final Object message) throws Exception {
		if(message instanceof SentenceMessage) {
			scala.concurrent.Future<SentenceMessage> f = Futures.future(new Callable<SentenceMessage>() {

				@Override
				public SentenceMessage call() throws Exception {
					SentenceMessage m2 = (SentenceMessage) message;
					vec.processSentence(m2.getSentence());
					return m2;
				}
				
			},context().dispatcher());
			
			f.onComplete(new OnComplete<SentenceMessage>() {

				@Override
				public void onComplete(Throwable arg0, SentenceMessage m2)
						throws Throwable {

					log.info("Processed sentence");
					m2.getChanged().set(System.currentTimeMillis());
										
				}
				
			}, context().dispatcher());
			
		}
		
		else 
			unhandled(message);
		
		
	}

	

	
	public static class SentenceActorCreator implements Creator<SentenceActor> {
		
		private Word2Vec vec;
		public SentenceActorCreator(Word2Vec vec) {
			this.vec = vec;
		}
		
		
		@Override
		public SentenceActor create() throws Exception {
			return new SentenceActor(vec);
		}
		
	}
}
