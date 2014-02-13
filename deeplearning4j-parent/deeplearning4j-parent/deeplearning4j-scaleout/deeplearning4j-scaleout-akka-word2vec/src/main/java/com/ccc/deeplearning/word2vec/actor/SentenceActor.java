package com.ccc.deeplearning.word2vec.actor;

import com.ccc.deeplearning.word2vec.Word2Vec;

import akka.actor.UntypedActor;
import akka.japi.Creator;

public class SentenceActor extends UntypedActor {

	private Word2Vec vec;

	
	public SentenceActor(Word2Vec vec) {
		super();
		this.vec = vec;
	}




	@Override
	public void onReceive(Object message) throws Exception {
		if(message instanceof SentenceMessage) {
			SentenceMessage m2 = (SentenceMessage) message;
			vec.processSentence(m2.getSentence(), m2.getCounter());
			m2.getChanged().set(System.currentTimeMillis());
			
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
