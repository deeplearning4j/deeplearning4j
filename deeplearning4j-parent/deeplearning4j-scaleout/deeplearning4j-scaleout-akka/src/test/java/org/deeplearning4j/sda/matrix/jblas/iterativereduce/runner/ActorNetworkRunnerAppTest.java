package org.deeplearning4j.sda.matrix.jblas.iterativereduce.runner;

import static org.junit.Assert.assertEquals;

import java.io.IOException;

import org.apache.curator.test.TestingServer;
import org.deeplearning4j.matrix.jblas.iterativereduce.actor.multilayer.ActorNetworkRunnerApp;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.zookeeper.ZookeeperConfigurationRetriever;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class ActorNetworkRunnerAppTest {
	
	private TestingServer server;
	private static Logger log = LoggerFactory.getLogger(ActorNetworkRunnerAppTest.class);
	
	
	@Before
	public void init() throws Exception {
		server = new TestingServer(2181);
	}
	
	
	@Test
	public void testConfig() {
		ActorNetworkRunnerApp app = new ActorNetworkRunnerApp(new String[] {
				"-data","mnist","-a","sda","-i","1","-o","1"
		});
		
		assertEquals("mnist",app.getData());
		assertEquals("sda",app.getAlgorithm());
		assertEquals("1",String.valueOf(app.getInputs()));
		assertEquals("1",String.valueOf(app.getOutputs()));
		
	}

	@Test
	public void testDataSetFetch() throws Exception {
		ActorNetworkRunnerApp app = new ActorNetworkRunnerApp(new String[]{
				"-data","mnist","-a","sda","-i","1","-o","1"
		});
		
		app.exec();
		app.train();
		
		ZookeeperConfigurationRetriever retriever = new ZookeeperConfigurationRetriever("master");

		Conf conf = retriever.retreive();
		assertEquals(true,conf != null);
		while(!app.isDone()) {
			Thread.sleep(10000);
		}
		log.info("Done");
		
	}
	
	@After
	public void after() throws IOException {
		server.close();
	}
}
