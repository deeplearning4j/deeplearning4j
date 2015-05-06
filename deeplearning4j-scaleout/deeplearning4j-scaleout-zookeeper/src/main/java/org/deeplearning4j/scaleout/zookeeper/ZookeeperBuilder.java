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

import java.io.IOException;

import org.apache.zookeeper.WatchedEvent;
import org.apache.zookeeper.Watcher;
import org.apache.zookeeper.Watcher.Event.KeeperState;
import org.apache.zookeeper.ZooKeeper;
import org.apache.zookeeper.ZooKeeper.States;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
/**
 * ZooKeeper client builder with default host of local host, port 2181, and timeout of 1000
 * @author Adam Gibson
 *
 */
public class ZookeeperBuilder implements Watcher {
	private String host;
	private int port;
	private int timeout;
	private Watcher watcher;
	private ZooKeeper keeper;
	private static final Logger log = LoggerFactory.getLogger(ZookeeperBuilder.class);
	public ZookeeperBuilder() {
		host = "localhost";
		port = 2181;
		timeout = 10000;

	}

	public ZooKeeper build() {
		try {
			keeper = new ZooKeeper(host + ":" + port,timeout,this);
			while(keeper.getState() != States.CONNECTED) {
				Thread.sleep(15000);
				log.info("Waiting to connect to zookeeper");
			}
			return keeper;
		} catch (IOException e) {
			throw new RuntimeException(e);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			return null;
		}

	}


	public ZookeeperBuilder setWatcher(Watcher watcher) {
		this.watcher = watcher;
		return this;
	}

	public ZookeeperBuilder setPort(int port) {
		this.port = port;
		return this;
	}

	public ZookeeperBuilder setHost(String host) {
		this.host = host;
		return this;
	}

	public ZookeeperBuilder setSessionTimeout(int timeout) {
		this.timeout = timeout;
		return this;
	}

	@Override
	public void process(WatchedEvent event) {
		if(event.getState() == KeeperState.SyncConnected) {
			log.info("Synced");	
		}
		else if(event.getState() == KeeperState.Disconnected) {
			keeper = build();
		}
		else if(event.getState() == KeeperState.Expired) {
			keeper = build();
		}
		if(watcher!=null)
			watcher.process(event);
		log.info("Processed event...");
	}




}
