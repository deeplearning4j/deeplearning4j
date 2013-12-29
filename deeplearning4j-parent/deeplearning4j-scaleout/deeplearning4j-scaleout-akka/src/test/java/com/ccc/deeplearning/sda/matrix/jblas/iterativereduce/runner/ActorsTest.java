package com.ccc.deeplearning.sda.matrix.jblas.iterativereduce.runner;

import static org.junit.Assert.assertEquals;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import akka.actor.ActorSystem;
import akka.actor.Props;
import akka.testkit.JavaTestKit;
import akka.testkit.TestActorRef;

import com.ccc.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.core.ResetMessage;
import com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.core.actor.BatchActor;
import com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.multilayer.MasterActor;
import com.ccc.deeplearning.scaleout.conf.Conf;
import com.ccc.deeplearning.scaleout.iterativereduce.UpdateableImpl;

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
		final TestActorRef<BatchActor> ref = TestActorRef.create(system, Props.create(new BatchActor.BatchActorFactory(new MnistDataSetIterator(1,1))), "testA");
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
		final TestActorRef<BatchActor> ref1 = TestActorRef.create(system, Props.create(new BatchActor.BatchActorFactory(new MnistDataSetIterator(1,1))), "testC");
		BatchActor ref2 = ref1.underlyingActor();
		Conf c = new Conf();
		c.put(Conf.PRE_TRAIN_EPOCHS, 1);
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
