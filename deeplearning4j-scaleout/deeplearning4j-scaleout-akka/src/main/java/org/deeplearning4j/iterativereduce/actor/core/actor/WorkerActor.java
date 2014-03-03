package org.deeplearning4j.iterativereduce.actor.core.actor;

import java.util.List;

import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.iterativereduce.ComputableWorker;
import org.deeplearning4j.scaleout.iterativereduce.Updateable;
import org.jblas.DoubleMatrix;

import scala.concurrent.duration.Duration;
import akka.actor.ActorRef;
import akka.actor.OneForOneStrategy;
import akka.actor.SupervisorStrategy;
import akka.actor.SupervisorStrategy.Directive;
import akka.actor.UntypedActor;
import akka.cluster.Cluster;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.contrib.pattern.DistributedPubSubMediator.Put;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import akka.japi.Function;


public abstract class WorkerActor<E extends Updateable<?>> extends UntypedActor implements DeepLearningConfigurable,ComputableWorker<E> {
	protected DoubleMatrix combinedInput,outcomes;

	protected ActorRef mediator = DistributedPubSubExtension.get(getContext().system()).mediator();
	protected E e;
	protected E results;
	protected LoggingAdapter log = Logging.getLogger(getContext().system(), this);
	protected int fineTuneEpochs;
	protected int preTrainEpochs;
	protected int[] hiddenLayerSizes;
	protected int numHidden;
	protected int numVisible;
	protected int numHiddenNeurons;
	protected int renderWeightEpochs;
	protected long seed;
	protected double learningRate;
	protected double corruptionLevel;
	protected Object[] extraParams;
	protected boolean useRegularization;
	Cluster cluster = Cluster.get(getContext().system());

	public final static String SYSTEM_NAME = "Workers";

	public WorkerActor(Conf conf) {
		setup(conf);
		//subscribe to broadcasts from workers (location agnostic)
		mediator.tell(new Put(getSelf()), getSelf());

		//subscribe to broadcasts from master (location agnostic)
		mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.BROADCAST, getSelf()), getSelf());
		//subscribe to shutdown messages
		mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.SHUTDOWN, getSelf()), getSelf());
		
	}


	@Override
	public void postStop() throws Exception {
		super.postStop();
		log.info("Post stop on worker actor");
		cluster.unsubscribe(getSelf());
	}
	
	@Override
	public void preStart() throws Exception {
		super.preStart();
		log.info("Pre start on worker");
	}



	@Override
	public E compute(List<E> records) {
		return compute();
	}

	@Override
	public abstract E compute();

	@Override
	public boolean incrementIteration() {
		return false;
	}

	@Override
	public void setup(Conf conf) {
		hiddenLayerSizes = conf.getLayerSizes();
		numHidden = conf.getnOut();
		numVisible = conf.getnIn();
		numHiddenNeurons = hiddenLayerSizes.length;
		seed = conf.getSeed();
		renderWeightEpochs = conf.getRenderWeightEpochs();
		useRegularization = conf.isUseRegularization();
		learningRate = conf.getPretrainLearningRate();
		preTrainEpochs = conf.getPretrainEpochs();
		fineTuneEpochs = conf.getFinetuneEpochs();
		corruptionLevel = conf.getCorruptionLevel();
		extraParams = conf.getDeepLearningParams();
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

	
	@Override
	public E getResults() {
		return results;
	}

	@Override
	public void update(E t) {
		this.e = t;
	}


	public synchronized DoubleMatrix getCombinedInput() {
		return combinedInput;
	}




	public synchronized void setCombinedInput(DoubleMatrix combinedInput) {
		this.combinedInput = combinedInput;
	}




	public synchronized DoubleMatrix getOutcomes() {
		return outcomes;
	}




	public synchronized void setOutcomes(DoubleMatrix outcomes) {
		this.outcomes = outcomes;
	}




	public synchronized ActorRef getMediator() {
		return mediator;
	}




	public synchronized void setMediator(ActorRef mediator) {
		this.mediator = mediator;
	}




	public synchronized E getE() {
		return e;
	}




	public synchronized void setE(E e) {
		this.e = e;
	}




	public synchronized int getFineTuneEpochs() {
		return fineTuneEpochs;
	}




	public synchronized void setFineTuneEpochs(int fineTuneEpochs) {
		this.fineTuneEpochs = fineTuneEpochs;
	}




	public synchronized int getPreTrainEpochs() {
		return preTrainEpochs;
	}




	public synchronized void setPreTrainEpochs(int preTrainEpochs) {
		this.preTrainEpochs = preTrainEpochs;
	}




	public synchronized int[] getHiddenLayerSizes() {
		return hiddenLayerSizes;
	}




	public synchronized void setHiddenLayerSizes(int[] hiddenLayerSizes) {
		this.hiddenLayerSizes = hiddenLayerSizes;
	}




	public synchronized int getNumHidden() {
		return numHidden;
	}




	public synchronized void setNumHidden(int numHidden) {
		this.numHidden = numHidden;
	}




	public synchronized int getNumVisible() {
		return numVisible;
	}




	public synchronized void setNumVisible(int numVisible) {
		this.numVisible = numVisible;
	}




	public synchronized int getNumHiddenNeurons() {
		return numHiddenNeurons;
	}




	public synchronized void setNumHiddenNeurons(int numHiddenNeurons) {
		this.numHiddenNeurons = numHiddenNeurons;
	}




	public synchronized long getSeed() {
		return seed;
	}




	public synchronized void setSeed(long seed) {
		this.seed = seed;
	}




	public synchronized double getLearningRate() {
		return learningRate;
	}




	public synchronized void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}




	public synchronized double getCorruptionLevel() {
		return corruptionLevel;
	}




	public synchronized void setCorruptionLevel(double corruptionLevel) {
		this.corruptionLevel = corruptionLevel;
	}




	public synchronized Object[] getExtraParams() {
		return extraParams;
	}




	public synchronized void setExtraParams(Object[] extraParams) {
		this.extraParams = extraParams;
	}




	public synchronized void setResults(E results) {
		this.results = results;
	}



}
