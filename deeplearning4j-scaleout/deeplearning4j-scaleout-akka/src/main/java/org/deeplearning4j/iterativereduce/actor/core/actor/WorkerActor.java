package org.deeplearning4j.iterativereduce.actor.core.actor;

import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

import org.deeplearning4j.iterativereduce.actor.core.ClearWorker;
import org.deeplearning4j.iterativereduce.actor.core.Job;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.iterativereduce.ComputableWorker;
import org.deeplearning4j.scaleout.iterativereduce.Updateable;
import org.jblas.DoubleMatrix;

import scala.concurrent.duration.Duration;
import akka.actor.ActorRef;
import akka.actor.Address;
import akka.actor.AddressFromURIString;
import akka.actor.OneForOneStrategy;
import akka.actor.SupervisorStrategy;
import akka.actor.SupervisorStrategy.Directive;
import akka.actor.UntypedActor;
import akka.cluster.Cluster;
import akka.cluster.ClusterEvent.MemberEvent;
import akka.contrib.pattern.ClusterReceptionistExtension;
import akka.contrib.pattern.DistributedPubSubExtension;
import akka.contrib.pattern.DistributedPubSubMediator;
import akka.contrib.pattern.DistributedPubSubMediator.Put;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import akka.japi.Function;

/**
 * Baseline worker actor class
 * @author Adam Gibson
 *
 * @param <E>
 */
public abstract class WorkerActor<E extends Updateable<?>> extends UntypedActor implements DeepLearningConfigurable,ComputableWorker<E> {
	protected DoubleMatrix combinedInput,outcomes;

	protected ActorRef mediator;
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
	protected String id;
	protected AtomicReference<Job> current;
	protected boolean useRegularization;
	Cluster cluster = Cluster.get(getContext().system());
	protected ActorRef clusterClient;
	public final static String SYSTEM_NAME = "Workers";
	protected String masterPath;
	ClusterReceptionistExtension receptionist = ClusterReceptionistExtension.get (getContext().system());

	public WorkerActor(Conf conf) {
		this(conf,null);
	}

	public WorkerActor(Conf conf,ActorRef client) {
		setup(conf);

		this.current = new AtomicReference<>(null);
		//subscribe to broadcasts from workers (location agnostic)
		mediator.tell(new Put(getSelf()), getSelf());

		//subscribe to broadcasts from master (location agnostic)
		mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.BROADCAST, getSelf()), getSelf());
		//subscribe to shutdown messages
		mediator.tell(new DistributedPubSubMediator.Subscribe(MasterActor.SHUTDOWN, getSelf()), getSelf());
		id = generateId();
		//replicate the network
		mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
				register()), getSelf());

		this.clusterClient = client;

		masterPath = conf.getMasterAbsPath();
		log.info("Registered with master " + id + " at master " + conf.getMasterAbsPath());
	}

	public WorkerState register() {
		return new WorkerState(this.id,getSelf());
	}

	public String generateId() {
		return UUID.randomUUID().toString();

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
		cluster.subscribe(getSelf(), MemberEvent.class);
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
		String url = conf.getMasterUrl();
		this.masterPath = conf.getMasterAbsPath();
		Address a = AddressFromURIString.apply(url);
		Cluster.get(context().system()).join(a);

		mediator = DistributedPubSubExtension.get(getContext().system()).mediator();

		availableForWork();
	}



	@Override
	public SupervisorStrategy supervisorStrategy() {
		return new OneForOneStrategy(0, Duration.Zero(),
				new Function<Throwable, Directive>() {
			public Directive apply(Throwable cause) {
				log.error("Problem with processing",cause);
				mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
						new ClearWorker(id)), getSelf());
				return SupervisorStrategy.restart();
			}
		});
	}

	/**
	 * Flags this worker as available to the master
	 */
	public void availableForWork() {
		log.info("Flagging availability of self " + id + " as available");
		//replicate the network
		mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
				new WorkerState(id,getSelf())), getSelf());
		
		Job j = new Job(id, null,true);
		j.setDone(true);
		mediator.tell(new DistributedPubSubMediator.Publish(MasterActor.MASTER,
				j), getSelf());
	
	}

	@Override
	public E getResults() {
		return results;
	}

	@Override
	public void update(E t) {
		this.e = t;
	}


	public  DoubleMatrix getCombinedInput() {
		return combinedInput;
	}




	public  void setCombinedInput(DoubleMatrix combinedInput) {
		this.combinedInput = combinedInput;
	}




	public  DoubleMatrix getOutcomes() {
		return outcomes;
	}




	public  void setOutcomes(DoubleMatrix outcomes) {
		this.outcomes = outcomes;
	}




	public  ActorRef getMediator() {
		return mediator;
	}




	public  void setMediator(ActorRef mediator) {
		this.mediator = mediator;
	}




	public  E getE() {
		return e;
	}




	public  void setE(E e) {
		this.e = e;
	}




	public  int getFineTuneEpochs() {
		return fineTuneEpochs;
	}




	public  void setFineTuneEpochs(int fineTuneEpochs) {
		this.fineTuneEpochs = fineTuneEpochs;
	}




	public  int getPreTrainEpochs() {
		return preTrainEpochs;
	}




	public  void setPreTrainEpochs(int preTrainEpochs) {
		this.preTrainEpochs = preTrainEpochs;
	}




	public  int[] getHiddenLayerSizes() {
		return hiddenLayerSizes;
	}




	public  void setHiddenLayerSizes(int[] hiddenLayerSizes) {
		this.hiddenLayerSizes = hiddenLayerSizes;
	}




	public  int getNumHidden() {
		return numHidden;
	}




	public  void setNumHidden(int numHidden) {
		this.numHidden = numHidden;
	}




	public  int getNumVisible() {
		return numVisible;
	}




	public  void setNumVisible(int numVisible) {
		this.numVisible = numVisible;
	}




	public  int getNumHiddenNeurons() {
		return numHiddenNeurons;
	}




	public  void setNumHiddenNeurons(int numHiddenNeurons) {
		this.numHiddenNeurons = numHiddenNeurons;
	}




	public  long getSeed() {
		return seed;
	}




	public  void setSeed(long seed) {
		this.seed = seed;
	}




	public  double getLearningRate() {
		return learningRate;
	}




	public  void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}




	public  double getCorruptionLevel() {
		return corruptionLevel;
	}




	public  void setCorruptionLevel(double corruptionLevel) {
		this.corruptionLevel = corruptionLevel;
	}




	public  Object[] getExtraParams() {
		return extraParams;
	}




	public  void setExtraParams(Object[] extraParams) {
		this.extraParams = extraParams;
	}




	public  void setResults(E results) {
		this.results = results;
	}

	public  Job getCurrent() {
		return this.current.get();
	}

	public  void setCurrent(Job current) {
		this.current.set(current);
	}

	/**
	 * Clears the current job
	 */
	protected  void clearCurrentJob() {
		setCurrent(null);
	}


}
