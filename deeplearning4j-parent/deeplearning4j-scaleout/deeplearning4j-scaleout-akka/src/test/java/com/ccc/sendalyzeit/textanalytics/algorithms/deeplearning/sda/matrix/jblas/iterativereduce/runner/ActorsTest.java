package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.runner;

import java.io.IOException;

import static org.junit.Assert.*;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator.impl.MnistDataSetIterator;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.actor.BatchActor;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.actor.MasterActor;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.actor.ResetMessage;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.Conf;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.jblas.UpdateableMatrix;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import akka.testkit.JavaTestKit;
import akka.testkit.TestActorRef;

public class ActorsTest {

	static ActorSystem system;

	@BeforeClass
	public static void setup() {
		system = ActorSystem.create();
	}

	@AfterClass
	public static void teardown() {
		JavaTestKit.shutdownActorSystem(system);
		system = null;
	}

	@Test
	public void testBatchActor() throws Exception {
		final TestActorRef<BatchActor> ref = TestActorRef.create(system, Props.create(new BatchActor.BatchActorFactory(new MnistDataSetIterator(1))), "testA");
		BatchActor ref2 = ref.underlyingActor();
		assertEquals(true,ref2.getIter().hasNext());
		assertEquals(1,ref2.getIter().batch());
		ref2.onReceive(new ResetMessage());
		assertEquals(true,ref2.getIter().hasNext());
		assertEquals(1,ref2.getIter().batch());
		ref2.onReceive(1);
		assertEquals(true,ref2.getIter().hasNext());


	}

	@Test
	public void testMasterActor() throws Exception {
		final TestActorRef<BatchActor> ref1 = TestActorRef.create(system, Props.create(new BatchActor.BatchActorFactory(new MnistDataSetIterator(1))), "testC");
		BatchActor ref2 = ref1.underlyingActor();
		Conf c = new Conf();
		c.put(Conf.PRE_TRAIN_EPOCHS, 1);
		final TestActorRef<MasterActor> ref = TestActorRef.create(system, Props.create(new MasterActor.MasterActorFactory(c, ref1)), "testB");
		MasterActor master = ref.underlyingActor();
		assertEquals(c,master.getConf());
		assertEquals(0,master.getEpochsComplete());
		UpdateableMatrix m = master.getMasterMatrix();
		master.onReceive(m);
		//when an epoch occurs a reset is incurred and the number of epochs increases
		assertEquals(1,master.getEpochsComplete());
		assertEquals(1,ref2.getNumTimesReset());
		assertEquals(true,master.isDone());
	}


}
