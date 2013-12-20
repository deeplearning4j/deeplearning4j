package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.actor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.Conf;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.DeepLearningConfigurable;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.jblas.UpdateableMatrix;

public class ActorNetworkRunner implements DeepLearningConfigurable,EpochDoneListener {

	private ActorSystem system;
	private ActorRef masterActor;
	private int currEpochs = 0;
	private int epochs;
	private List<Pair<DoubleMatrix,DoubleMatrix>> samples;
	private UpdateableMatrix result;
	private CountDownLatch latch;
	private static Logger log = LoggerFactory.getLogger(ActorNetworkRunner.class);
	
	@Override
	public void setup(Conf conf) {
		system = ActorSystem.create();
		epochs = conf.getInt(PRE_TRAIN_EPOCHS);
		latch = new CountDownLatch(epochs);
		masterActor = system.actorOf(Props.create(new MasterActor.MasterActorFactory(conf)));
		masterActor.tell(this,masterActor);
		log.info("Setup master with epochs " + epochs);
	}
	
	public void train(List<Pair<DoubleMatrix,DoubleMatrix>> list) {
		this.samples = list;
		masterActor.tell(list,masterActor);
		try {
			latch.await();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
	}
	
	public void train(Pair<DoubleMatrix,DoubleMatrix> input) {
		this.samples = new ArrayList<>(Arrays.asList(input));
		masterActor.tell(input, masterActor);
		log.info("Training....");
		try {
			latch.await();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
		
	}
	
	public void train(DoubleMatrix input,DoubleMatrix labels) {
		train(new Pair<DoubleMatrix,DoubleMatrix>(input,labels));
		try {
			latch.await();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
	}
	
	public void shutdown() {
		system.shutdown();
	}

	@Override
	public void epochComplete(UpdateableMatrix result) {
		currEpochs++;
		if(currEpochs < epochs) {
			masterActor.tell(samples, masterActor);
			log.info("Updating result on epoch " + currEpochs);
		}
		this.result = result;
		latch.countDown();

	}

	public UpdateableMatrix getResult() {
		return result;
	}
	
	

}
