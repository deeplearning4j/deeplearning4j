package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.actor;

import java.io.DataOutputStream;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Queue;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.concurrent.duration.Duration;
import akka.actor.ActorRef;
import akka.actor.OneForOneStrategy;
import akka.actor.Props;
import akka.actor.SupervisorStrategy;
import akka.actor.SupervisorStrategy.Directive;
import akka.actor.UntypedActor;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.contrib.pattern.DistributedPubSubMediator.Put;
import akka.japi.Creator;
import akka.japi.Function;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.DeepLearningAccumulator;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.Conf;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.DeepLearningConfigurable;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.ComputableMaster;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.jblas.UpdateableMatrix;
import com.google.common.collect.Lists;

/**
 * Handles a set of workers and acts as a parameter server for iterative reduce
 * @author Adam Gibson
 *
 */
public class MasterActor extends UntypedActor implements DeepLearningConfigurable,ComputableMaster<UpdateableMatrix> {

	private Conf conf;
	private static Logger log = LoggerFactory.getLogger(MasterActor.class);
	protected UpdateableMatrix masterMatrix;
	private List<UpdateableMatrix> updates = new ArrayList<UpdateableMatrix>();
	private EpochDoneListener listener;
	private ActorRef batchActor;
	private int epochsComplete;
	private final ActorRef mediator = DistributedPubSubExtension.get(getContext().system()).mediator();
	public static String BROADCAST = "broadcast";
	public static String RESULT = "result";
	//number of batches over time
	private int partition;


	/**
	 * Creates the master and the workers with this given conf
	 * @param conf the neural net config to use
	 */
	public MasterActor(Conf conf,ActorRef batchActor) {
		this.conf = conf;
		this.batchActor = batchActor;
		//subscribe to broadcasts from workers (location agnostic)
	    mediator.tell(new Put(getSelf()), getSelf());

		mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.RESULT, getSelf()), getSelf());
		setup(conf);
		
		
	}

	public static Props propsFor(Conf conf,ActorRef batchActor) {
		return Props.create(new MasterActor.MasterActorFactory(conf,batchActor));
	}



	@Override
	public UpdateableMatrix compute(Collection<UpdateableMatrix> workerUpdates,
			Collection<UpdateableMatrix> masterUpdates) {


		DeepLearningAccumulator acc = new DeepLearningAccumulator();
		for(UpdateableMatrix m : workerUpdates) 
			acc.accumulate(m.get());

		masterMatrix.set(acc.averaged());

		return masterMatrix;
	}

	@Override
	public void setup(Conf conf) {
		//use the rng with the given seed
		RandomGenerator rng =  new MersenneTwister(conf.getLong(SEED));
		BaseMultiLayerNetwork matrix = new BaseMultiLayerNetwork.Builder<>()
				.numberOfInputs(conf.getInt(N_IN)).numberOfOutPuts(conf.getInt(OUT)).withClazz(conf.getClazz(CLASS))
				.hiddenLayerSizes(conf.getIntsWithSeparator(LAYER_SIZES, ",")).withRng(rng)
				.build();
		masterMatrix = new UpdateableMatrix(matrix);

	}


	@SuppressWarnings({ "unchecked" })
	@Override
	public void onReceive(Object message) throws Exception {
		if (message instanceof DistributedPubSubMediator.SubscribeAck) {
			DistributedPubSubMediator.SubscribeAck ack = (DistributedPubSubMediator.SubscribeAck) message;
			log.info("Subscribed " + ack.toString());
		}
		else if(message instanceof EpochDoneListener) {
			listener = (EpochDoneListener) message;
			log.info("Set listener");
		}

		else if(message instanceof UpdateableMatrix) {
			UpdateableMatrix up = (UpdateableMatrix) message;
			updates.add(up);
			if(updates.size() == partition) {
				masterMatrix = this.compute(updates, updates);
				if(listener != null)
					listener.epochComplete(masterMatrix);
				batchActor.tell(new ResetMessage(), getSelf());
				//TODO: there is a better way to to this.
				epochsComplete++;
				batchActor.tell(up, getSelf());
				updates.clear();
			}

		}

		//broadcast new weights to workers
		else if(message instanceof UpdateMessage) {
			mediator.tell(new DistributedPubSubMediator.Publish(BROADCAST,
					message), getSelf());
		}


		//list of examples
		else if(message instanceof List || message instanceof Pair) {

			if(message instanceof List) {
				List<Pair<DoubleMatrix,DoubleMatrix>> list = (List<Pair<DoubleMatrix,DoubleMatrix>>) message;
				//each pair in the matrix pairs maybe multiple rows
				splitListIntoRows(list);
				//delegate split to workers
				sendToWorkers(list);

			}

			//ensure split then send to workers
			else if(message instanceof Pair) {
				Pair<DoubleMatrix,DoubleMatrix> pair = (Pair<DoubleMatrix,DoubleMatrix>) message;

				//split pair up in to rows to ensure parallelism
				List<DoubleMatrix> inputs = pair.getFirst().rowsAsList();
				List<DoubleMatrix> labels = pair.getSecond().rowsAsList();

				List<Pair<DoubleMatrix,DoubleMatrix>> pairs = new ArrayList<>();
				for(int i = 0; i < inputs.size(); i++) {
					pairs.add(new Pair<>(inputs.get(i),labels.get(i)));
				}


				sendToWorkers(pairs);

			}
		}

		else
			unhandled(message);
	}


	private void sendToWorkers(List<Pair<DoubleMatrix,DoubleMatrix>> pairs) {
		int split = conf.getInt(SPLIT);
		List<List<Pair<DoubleMatrix,DoubleMatrix>>> splitList = Lists.partition(pairs, split);
		partition = splitList.size();

		for(int i = 0; i < splitList.size(); i++) 
			mediator.tell(new DistributedPubSubMediator.Publish(BROADCAST,
					new ArrayList<>(splitList.get(i))), getSelf());

	}



	private void splitListIntoRows(List<Pair<DoubleMatrix,DoubleMatrix>> list) {
		Queue<Pair<DoubleMatrix,DoubleMatrix>> q = new ArrayDeque<>(list);
		list.clear();
		log.info("Splitting list in to rows...");
		while(!q.isEmpty()) {
			Pair<DoubleMatrix,DoubleMatrix> pair = q.poll();
			List<DoubleMatrix> inputRows = pair.getFirst().rowsAsList();
			List<DoubleMatrix> labelRows = pair.getSecond().rowsAsList();
			for(int i = 0; i < inputRows.size(); i++) {
				list.add(new Pair<DoubleMatrix,DoubleMatrix>(inputRows.get(i),labelRows.get(i)));
			}
		}
	}


	public static class MasterActorFactory implements Creator<MasterActor> {

		public MasterActorFactory(Conf conf,ActorRef batchActor) {
			this.conf = conf;
			this.batchActor = batchActor;
		}

		private Conf conf;
		private ActorRef batchActor;
		/**
		 * 
		 */
		private static final long serialVersionUID = 1932205634961409897L;

		@Override
		public MasterActor create() throws Exception {
			return new MasterActor(conf,batchActor);
		}



	}

	@Override
	public void complete(DataOutputStream ds) {
		masterMatrix.get().write(ds);
	}

	@Override
	public UpdateableMatrix getResults() {
		return masterMatrix;
	}

	@Override
	public SupervisorStrategy supervisorStrategy() {
		return new OneForOneStrategy(0, Duration.Zero(),
				new Function<Throwable, Directive>() {
			public Directive apply(Throwable cause) {
				log.error("Problem with processing",cause);
				return SupervisorStrategy.stop();
			}
		});
	}
}
