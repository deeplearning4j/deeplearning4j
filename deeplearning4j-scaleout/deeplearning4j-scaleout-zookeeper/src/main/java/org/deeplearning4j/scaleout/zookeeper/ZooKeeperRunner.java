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

package org.deeplearning4j.scaleout.zookeeper;

import java.io.File;
import java.net.InetSocketAddress;

import org.apache.zookeeper.server.NIOServerCnxnFactory;
import org.apache.zookeeper.server.ZooKeeperServer;

/**
 * This is for running embedded zookeeper.
 * @author Adam Gibson
 *
 */
public class ZooKeeperRunner {
	
	
	private int clientPort = 2181; 
	private int numConnections = 5000;
	private int tickTime = 2000;
	private String dataDirectory = System.getProperty("java.io.tmpdir");

	public void run() throws Exception {
		
		File dir = new File(dataDirectory, "zookeeper").getAbsoluteFile();
		NIOServerCnxnFactory factory = new NIOServerCnxnFactory();
		
		ZooKeeperServer server = new ZooKeeperServer(dir, dir, tickTime);
		
		factory.setZooKeeperServer(server);
		factory.setMaxClientCnxnsPerHost(numConnections);
		factory.configure(new InetSocketAddress(clientPort), numConnections);
		factory.startup(server);
		
	}

	public  int getClientPort() {
		return clientPort;
	}

	public  void setClientPort(int clientPort) {
		this.clientPort = clientPort;
	}

	public  int getNumConnections() {
		return numConnections;
	}

	public  void setNumConnections(int numConnections) {
		this.numConnections = numConnections;
	}

	public  int getTickTime() {
		return tickTime;
	}

	public  void setTickTime(int tickTime) {
		this.tickTime = tickTime;
	}

	public  String getDataDirectory() {
		return dataDirectory;
	}

	public  void setDataDirectory(String dataDirectory) {
		this.dataDirectory = dataDirectory;
	}

	
}
