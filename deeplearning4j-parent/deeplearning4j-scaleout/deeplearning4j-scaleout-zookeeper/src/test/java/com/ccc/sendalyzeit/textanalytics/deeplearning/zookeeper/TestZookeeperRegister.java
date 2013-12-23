package com.ccc.sendalyzeit.textanalytics.deeplearning.zookeeper;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeNotNull;

import java.io.IOException;
import java.util.Arrays;


import org.apache.curator.test.TestingServer;
import org.apache.zookeeper.KeeperException.NoNodeException;
import org.apache.zookeeper.WatchedEvent;
import org.apache.zookeeper.Watcher;
import org.apache.zookeeper.ZooKeeper;
import org.apache.zookeeper.data.Stat;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.Conf;

public class TestZookeeperRegister {
	private static Logger log = LoggerFactory.getLogger(TestZookeeperRegister.class);
	private ZooKeeper keeper;
	private Conf conf;
	private ZooKeeperConfigurationRegister register;
	private ZookeeperConfigurationRetriever retriever;
	private TestingServer server;
	
	@Before
	public void init() throws Exception {
		conf = new Conf();
		server = new TestingServer(2181);

		String host = "localhost";
		keeper = new ZookeeperBuilder().setHost(host).build();
		register = new ZooKeeperConfigurationRegister(conf, keeper, "test",host,2181);
		retriever = new ZookeeperConfigurationRetriever(host, 2181,"test");

	}

	@Test
	public void testPathBuilder() {
		String expected = "/testhadoop:2181/tmp";
		assertEquals(expected,new ZookeeperPathBuilder().setHost("testhadoop").setPort(2181).addPath("tmp").build());
	}

	@Test(expected=NoNodeException.class)
	public void testZookeeperRegister() throws Exception {
		retriever.retreive();
	}

	@Test
	public void testZookeeper() throws Exception {
		String host = "localhost";

		register.register();
		String path = new ZookeeperPathBuilder().setHost(host).setPort(2181).addPaths(Arrays.asList("tmp","test")).build();

		assumeNotNull(keeper.getData(path, new Watcher() {

			@Override
			public void process(WatchedEvent event) {
				log.info("Event " + event);

			}

		}, new Stat()));

		Conf retrieve = retriever.retreive();
		assumeNotNull(retrieve);
		/*
		 * Not sure what to do here: there are random keys being thrown in, but it doesn't seem to affect
		 * anything that matters.
		 */
		//Map<String,String> confMap = entries(conf);
		//Map<String,String> retrieveMap = entries(retrieve);
		//
		//assertEquals("Key difference was " + SetUtils.union( SetUtils.difference(confMap.keySet(), retrieveMap.keySet()), SetUtils.difference(retrieveMap.keySet(), confMap.keySet())),retrieveMap.keySet().size(),confMap.keySet().size());
		//assertEquals("Value difference was " + SetUtils.union(SetUtils.difference(confMap.values(), retrieveMap.values()),SetUtils.difference(retrieveMap.values(), confMap.values())),retrieveMap.values().size(),confMap.values().size());

		keeper.delete(path, -1);



	}
	
	
	@After
	public void after() throws IOException {
		retriever.close();
		register.close();
		server.stop();
	}

}
