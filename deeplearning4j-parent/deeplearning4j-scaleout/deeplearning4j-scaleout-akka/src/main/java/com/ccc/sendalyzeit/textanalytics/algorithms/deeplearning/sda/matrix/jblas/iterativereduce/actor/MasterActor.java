package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.actor;

import java.io.DataOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.DeepLearningAccumulator;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.Conf;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.DeepLearningConfigurable;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.ComputableMaster;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.jblas.UpdateableMatrix;
import com.google.common.collect.Lists;

import akka.actor.ActorRef;
import akka.actor.Props;
import akka.actor.UntypedActor;
import akka.japi.Creator;

public class MasterActor extends UntypedActor implements DeepLearningConfigurable,ComputableMaster<UpdateableMatrix> {

	private Conf conf;
	private static Logger log = LoggerFactory.getLogger(MasterActor.class);
	protected UpdateableMatrix masterMatrix;
	private List<ActorRef> workers = new ArrayList<ActorRef>();
	private List<UpdateableMatrix> updates = new ArrayList<UpdateableMatrix>();
	private EpochDoneListener listener;

	public MasterActor(Conf conf) {
		this.conf = conf;
		setup(conf);
		int split = conf.getInt(SPLIT);
		for(int i = 0; i < split; i++) {
			Conf c = conf.copy();
			c.put(FINE_TUNE_EPOCHS, 1);
			c.put(PRE_TRAIN_EPOCHS,1);
			workers.add(context().actorOf(Props.create(new WorkerActor.WorkerActorFactory(c))));
		}
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
		if(message instanceof EpochDoneListener) {
			listener = (EpochDoneListener) message;
			log.info("Set listener");
		}
		else if(message instanceof UpdateableMatrix) {
			UpdateableMatrix up = (UpdateableMatrix) message;
			updates.add(up);
			if(updates.size() == workers.size()) {
				masterMatrix = this.compute(updates, updates);
				if(listener != null)
					listener.epochComplete(masterMatrix);
				updates.clear();
			}
			
		}

		else if(message instanceof UpdateMessage) {
			for(ActorRef worker : workers) {
				worker.tell(message, getSelf());
			}
		}


		//list of examples
		else if(message instanceof List || message instanceof Pair) {
			if(message instanceof List) {
				List<Pair<DoubleMatrix,DoubleMatrix>> list = (List<Pair<DoubleMatrix,DoubleMatrix>>) message;
				int split = conf.getInt(SPLIT);
				List<List<Pair<DoubleMatrix,DoubleMatrix>>> splitList = Lists.partition(list, split);
				for(int i = 0; i < splitList.size(); i++) {
					workers.get(i).tell(splitList.get(i), getSelf());
				}

			}
			else if(message instanceof Pair) {
				Pair<DoubleMatrix,DoubleMatrix> pair = (Pair<DoubleMatrix,DoubleMatrix>) message;
				List<DoubleMatrix> inputs = pair.getFirst().rowsAsList();
				List<DoubleMatrix> labels = pair.getSecond().rowsAsList();
				List<Pair<DoubleMatrix,DoubleMatrix>> pairs = new ArrayList<>();
				for(int i = 0; i < inputs.size(); i++) {
					pairs.add(new Pair<>(inputs.get(i),labels.get(i)));
				}

				int split = conf.getInt(SPLIT);
				List<List<Pair<DoubleMatrix,DoubleMatrix>>> splitList = Lists.partition(pairs, split);
				if(splitList.size() >= workers.size()) {
					log.info("Adding workers to accomadate split and load...");
					while(workers.size() < splitList.size()) {
						Conf c = conf.copy();
						c.put(FINE_TUNE_EPOCHS, 1);
						c.put(PRE_TRAIN_EPOCHS,1);
						workers.add(context().actorOf(Props.create(new WorkerActor.WorkerActorFactory(c))));

					}
				}

				for(int i = 0; i < splitList.size(); i++) {
					workers.get(i).tell(splitList.get(i), getSelf());
				}

			}
		}
		else
			unhandled(message);
	}


	public static class MasterActorFactory implements Creator<MasterActor> {

		public MasterActorFactory(Conf conf) {
			this.conf = conf;
		}

		private Conf conf;
		/**
		 * 
		 */
		private static final long serialVersionUID = 1932205634961409897L;

		@Override
		public MasterActor create() throws Exception {
			return new MasterActor(conf);
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

}
