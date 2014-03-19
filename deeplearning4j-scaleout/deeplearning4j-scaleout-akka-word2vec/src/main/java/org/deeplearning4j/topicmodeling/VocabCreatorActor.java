package org.deeplearning4j.topicmodeling;

import java.io.File;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;

import scala.concurrent.Future;

import akka.actor.UntypedActor;
import akka.dispatch.Futures;
import akka.dispatch.OnComplete;

public class VocabCreatorActor extends UntypedActor {

	
	private VocabCreator vocabCreator;
	private CountDownLatch latch;
	
	
	public VocabCreatorActor(VocabCreator vocabCreator,CountDownLatch latch) {
		super();
		this.vocabCreator = vocabCreator;
		this.latch = latch;
	}



	@Override
	public void onReceive(final Object message) throws Exception {
		if(message instanceof File) {
			Future<Void> f = Futures.future(new Callable<Void>() {

				@Override
				public Void call() throws Exception {
					File f = (File) message;
					vocabCreator.addForDoc(f);
					return null;
				}
				
			}, context().dispatcher());
			
			
			f.onComplete(new OnComplete<Void>() {

				@Override
				public void onComplete(Throwable arg0, Void arg1)
						throws Throwable {
					if(arg0 != null)
						throw arg0;
					latch.countDown();
				}
				
			}, context().dispatcher());
			
			
		}
		else {
			unhandled(message);
		}
	}

}
