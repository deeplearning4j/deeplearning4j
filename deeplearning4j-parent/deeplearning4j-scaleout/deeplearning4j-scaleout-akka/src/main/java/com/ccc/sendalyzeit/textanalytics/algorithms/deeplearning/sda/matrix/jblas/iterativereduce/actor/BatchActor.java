package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.actor;

import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator.DataSetIterator;

import akka.actor.ActorRef;
import akka.actor.UntypedActor;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.japi.Creator;

public class BatchActor extends UntypedActor {

	private DataSetIterator iter;
	private final ActorRef mediator = DistributedPubSubExtension.get(getContext().system()).mediator();
	private int numTimesReset;
	
	
	public BatchActor(DataSetIterator iter) {
		this.iter = iter;
	}
	
	
	@Override
	public void onReceive(Object arg0) throws Exception {
	    if(arg0 instanceof ResetMessage) {
	    	iter.reset();
	    	numTimesReset++;
	    }
		else if(iter.hasNext()) {
			//start the pipeline
			mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.RESULT,
					iter.next()), mediator);

		}
		else
			unhandled(arg0);
		
	}
	
	
	
	public DataSetIterator getIter() {
		return iter;
	}



	public static class BatchActorFactory implements Creator<BatchActor> {

		/**
		 * 
		 */
		private static final long serialVersionUID = -2260113511909990862L;

		public BatchActorFactory(DataSetIterator iter) {
			this.iter = iter;
		}
		
		private DataSetIterator iter;

		@Override
		public BatchActor create() throws Exception {
			return new BatchActor(iter);
		}
		
		
		
	}



	public int getNumTimesReset() {
		return numTimesReset;
	}
	
	
	
	

}
