/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.zookeeper;

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
import org.canova.api.conf.Configuration;
import org.deeplearning4j.scaleout.zookeeper.ZooKeeperConfigurationRegister;
import org.deeplearning4j.scaleout.zookeeper.ZookeeperBuilder;
import org.deeplearning4j.scaleout.zookeeper.ZookeeperConfigurationRetriever;
import org.deeplearning4j.scaleout.zookeeper.ZookeeperPathBuilder;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Ignore
public class TestZookeeperRegister {
	private static final Logger log = LoggerFactory.getLogger(TestZookeeperRegister.class);
	private ZooKeeper keeper;
	private Configuration conf;
	private ZooKeeperConfigurationRegister register;
	private ZookeeperConfigurationRetriever retriever;
	private TestingServer server;

	@Before
	public void init() throws Exception {
		conf = new Configuration();
		try {
			server = new TestingServer(2181);
		}catch(Exception e) {
			log.warn("Port already taken; using current zookeeper");
		}
		String host = "localhost";
		keeper = new ZookeeperBuilder().setHost(host).build();
		register = new ZooKeeperConfigurationRegister(conf, "test","localhost",2181);
		retriever = new ZookeeperConfigurationRetriever(host, 2181,"test");

	}

	@Test
	public void testPathBuilder() {
		String expected = "/testhadoop:2181/tmp";
		assertEquals(expected,new ZookeeperPathBuilder().setHost("testhadoop").setPort(2181).addPath("tmp").build());
	}

	@Test(expected=NoNodeException.class)
	public void testZookeeperRegister() throws Exception {
		retriever.retrieve();
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

        Configuration retrieve = retriever.retrieve();
		assumeNotNull(retrieve);

		keeper.delete(path, -1);



	}


	@After
	public void after() throws IOException {
		retriever.close();
		register.close();
		server.stop();
	}

}
