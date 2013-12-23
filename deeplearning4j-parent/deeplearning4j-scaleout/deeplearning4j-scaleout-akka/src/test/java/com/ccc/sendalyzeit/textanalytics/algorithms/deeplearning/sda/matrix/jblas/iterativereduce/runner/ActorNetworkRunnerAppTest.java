package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.runner;

import static org.junit.Assert.*;

import java.io.IOException;

import org.apache.curator.test.TestingServer;
import org.apache.zookeeper.ZooKeeper;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.actor.ActorNetworkRunnerApp;
import com.ccc.sendalyzeit.textanalytics.deeplearning.zookeeper.ZookeeperBuilder;
import com.ccc.sendalyzeit.textanalytics.deeplearning.zookeeper.ZookeeperConfigurationRetriever;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.Conf;

public class ActorNetworkRunnerAppTest {
	
	private TestingServer server;
	
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
		
		ZooKeeper zk = new ZookeeperBuilder().setHost("127.0.0.1").build();
		ZookeeperConfigurationRetriever retriever = new ZookeeperConfigurationRetriever(zk, "master");
		Conf conf = retriever.retreive();
		assertEquals(true,conf != null);
		
		app.shutdown();
		
	}
	
	@After
	public void after() throws IOException {
		server.close();
	}
}
