package org.deeplearning4j.sda.matrix.jblas.iterativereduce.runner;

import static org.junit.Assert.assertEquals;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.matrix.jblas.iterativereduce.actor.core.ResetMessage;
import org.deeplearning4j.matrix.jblas.iterativereduce.actor.core.actor.BatchActor;
import org.deeplearning4j.matrix.jblas.iterativereduce.actor.multilayer.MasterActor;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

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
		final TestActorRef<BatchActor> ref = TestActorRef.create(system, Props.create(new BatchActor.BatchActorFactory(new MnistDataSetIterator(1,1),1)), "testA");
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
		final TestActorRef<BatchActor> ref1 = TestActorRef.create(system, Props.create(new BatchActor.BatchActorFactory(new MnistDataSetIterator(1,1), 1)), "testC");
		BatchActor ref2 = ref1.underlyingActor();
		Conf c = new Conf();
		c.setPretrainEpochs(1);
		final TestActorRef<MasterActor> ref = TestActorRef.create(system, Props.create(new MasterActor.MasterActorFactory(c, ref1)), "testB");
		MasterActor master = ref.underlyingActor();
		assertEquals(c,master.getConf());
		assertEquals(0,master.getEpochsComplete());
		UpdateableImpl m = master.getMasterResults();
		master.onReceive(m);
		//when an epoch occurs a reset is incurred and the number of epochs increases
		assertEquals(1,master.getEpochsComplete());
		assertEquals(1,ref2.getNumTimesReset());
		assertEquals(true,master.isDone());
	}


}
