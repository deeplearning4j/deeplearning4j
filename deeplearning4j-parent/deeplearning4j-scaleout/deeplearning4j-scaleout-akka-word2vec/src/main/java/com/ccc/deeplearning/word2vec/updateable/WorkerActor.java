package com.ccc.deeplearning.word2vec.updateable;

import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.ActorRef;
import akka.actor.Props;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.contrib.pattern.DistributedPubSubMediator.Put;
import akka.japi.Creator;

import com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.core.UpdateMessage;
import com.ccc.deeplearning.nn.activation.HardTanh;
import com.ccc.deeplearning.scaleout.conf.Conf;
import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.nn.multilayer.Word2VecMultiLayerNetwork;
import com.ccc.deeplearning.word2vec.util.Window;

/**
 * Trains based on a list of windows
 * converting the inputs to matrices via word2vec
 * @author Adam Gibson
 *
 */
public class WorkerActor extends com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.core.actor.WorkerActor<Word2VecUpdateable> {
	private Word2VecMultiLayerNetwork network;
	private List<Window> input;
	private Word2Vec vec;
	protected Word2VecUpdateable workerMatrix;
	private ActorRef mediator = DistributedPubSubExtension.get(getContext().system()).mediator();
	private List<String> labels;
	private static Logger log = LoggerFactory.getLogger(WorkerActor.class);
	public final static String SYSTEM_NAME = "Workers";

	public WorkerActor(Conf conf,Word2Vec vec,List<String> labels) {
		super(conf);
		this.vec = vec;
		this.labels = labels;
		setup(conf);
		
		
		//subscribe to broadcasts from workers (location agnostic)
	    mediator.tell(new Put(getSelf()), getSelf());

		//subscribe to broadcasts from master (location agnostic)
		mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.BROADCAST, getSelf()), getSelf());
	}


	public static Props propsFor(ActorRef actor,Conf conf,Word2Vec vec,List<String> labels) {
		return Props.create(new WorkerActor.WorkerActorFactory(conf,vec,labels));
	}

	public static Props propsFor(Conf conf,Word2Vec vec,List<String> labels) {
		return Props.create(new WorkerActor.WorkerActorFactory(conf,vec,labels));
	}


	@SuppressWarnings("unchecked")
	@Override
	public void onReceive(Object message) throws Exception {
		if (message instanceof DistributedPubSubMediator.SubscribeAck) {
			DistributedPubSubMediator.SubscribeAck ack = (DistributedPubSubMediator.SubscribeAck) message;
			log.info("Subscribed to " + ack.toString());
		}
		else if(message instanceof List) {
			List<Window> input = (List<Window>) message;
			updateTraining(input);

		}

		else if(message instanceof UpdateMessage) {
			UpdateMessage<Word2VecUpdateable> m = (UpdateMessage<Word2VecUpdateable>) message;
			workerMatrix = (Word2VecUpdateable) m.getUpdateable().get();
		}
		else
			unhandled(message);
	}

	private void updateTraining(List<Window> windows) {
		this.input = windows;
		Word2VecUpdateable work = compute();
		log.info("Updating parent actor...");
		//update parameters in master param server
		mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.RESULT,
				work), getSelf());
	}

	@Override
	public Word2VecUpdateable compute(List<Word2VecUpdateable> records) {
		return compute();
	}

	@Override
	public Word2VecUpdateable compute() {
		log.info("Training network");
		network.trainNetwork(input, extraParams);
		return new Word2VecUpdateable(network);
	}

	@Override
	public boolean incrementIteration() {
		return false;
	}

	@Override
	public void setup(Conf conf) {
		super.setup(conf);
		
		RandomGenerator rng = new MersenneTwister(conf.getLong(SEED));
		network = new Word2VecMultiLayerNetwork.Builder().withWord2Vec(vec).withLabels(labels)
				.withActivation(new HardTanh())
				.numberOfInputs(numVisible).numberOfOutPuts(numHidden)
				.hiddenLayerSizes(hiddenLayerSizes).withRng(rng)
				.withClazz(conf.getClazz(CLASS)).build();
		
	}



	@Override
	public Word2VecUpdateable getResults() {
		return workerMatrix;
	}

	@Override
	public void update(Word2VecUpdateable t) {
		this.workerMatrix = t;
	}


	public static class WorkerActorFactory implements Creator<WorkerActor> {

		/**
		 * 
		 */
		private static final long serialVersionUID = 381253681712601968L;

		public WorkerActorFactory(Conf conf,Word2Vec vec,List<String> labels) {
			this.conf = conf;
			this.vec = vec;
			this.labels = labels;
		}

		private Conf conf;
		private Word2Vec vec;
		private List<String> labels;
		@Override
		public WorkerActor create() throws Exception {
			return new WorkerActor(conf,vec,labels);
		}

	}


}
