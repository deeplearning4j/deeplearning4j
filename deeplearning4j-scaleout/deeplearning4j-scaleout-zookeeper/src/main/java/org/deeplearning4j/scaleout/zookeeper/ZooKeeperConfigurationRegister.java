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

import java.util.Arrays;

import org.apache.zookeeper.CreateMode;
import org.apache.zookeeper.KeeperException;
import org.apache.zookeeper.Op;
import org.apache.zookeeper.WatchedEvent;
import org.apache.zookeeper.Watcher;
import org.apache.zookeeper.Watcher.Event.KeeperState;
import org.apache.zookeeper.ZooDefs;
import org.apache.zookeeper.ZooKeeper;
import org.apache.zookeeper.data.Stat;
import org.canova.api.conf.Configuration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.InetAddress;
import java.net.UnknownHostException;

/**
 * This registers a given hadoop configuration with a zookeeper cluster.
 * Configurations are serialized in the form of
 * key=value
 * key2=value2
 * This is meant for use by a map reduce cluster
 * to getFromOrigin run time configuration parameters.
 * Usage:
 * ZooKeeperConfigurationRegiste register = ...
 * register.register();
 * @author Adam Gibson
 *
 */
public class ZooKeeperConfigurationRegister implements Watcher {

	private Configuration configuration;
	private ZooKeeper zk;
	private String id;
	private String host;
	private int port;
	private static final Logger log = LoggerFactory.getLogger(ZooKeeperConfigurationRegister.class);

	/**
	 * 
	 * @param configuration the configuration to serialize
	 * @param id the job id to store metadata for
	 * @param host host of the zookeeper cluster (note this is also provided to help setup the zk directory structure)
	 * @param port the port of the zookeeper cluster (note this is also provided to help setup the zk directory structure)
	 */
	public ZooKeeperConfigurationRegister(Configuration configuration,String id,String host,int port) {
		super();
		this.configuration = configuration;
		this.id = id;
		this.host = host;
		this.port = port;
		if(zk ==null)
			this.zk = new ZookeeperBuilder().setHost(host).setPort(port).build();
	}


	public static byte[] serialize(Object obj) throws IOException {
		ByteArrayOutputStream b = new ByteArrayOutputStream();
		ObjectOutputStream o = new ObjectOutputStream(b);
		o.writeObject(obj);
		return b.toByteArray();
	}

	public static Object deserialize(byte[] bytes) throws IOException, ClassNotFoundException {
		ByteArrayInputStream b = new ByteArrayInputStream(bytes);
		ObjectInputStream o = new ObjectInputStream(b);
		return o.readObject();
	}


	/**
	 * Registers the configuration in zookeeper
	 */
	public void register() {
		byte[] data;
		try {
			data = serialize(configuration);
		} catch (IOException e1) {
			throw new RuntimeException(e1);
		}
		try {
			zk.multi(Arrays.asList(
					Op.create(new ZookeeperPathBuilder().setHost(host).setPort(port).build(), data, ZooDefs.Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT),
					Op.create(new ZookeeperPathBuilder().setHost(host).setPort(port).addPath("tmp").build(), data, ZooDefs.Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT)

					));
			log.info("Register at " + host);
			zk.multi(Arrays.asList(
					Op.create(new ZookeeperPathBuilder().setHost("127.0.0.1").setPort(port).build(), data, ZooDefs.Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT),
					Op.create(new ZookeeperPathBuilder().setHost("127.0.0.1").setPort(port).addPath("tmp").build(), data, ZooDefs.Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT)

					));
			log.info("Registered at 127.0.0.1");
			zk.multi(Arrays.asList(
					Op.create(new ZookeeperPathBuilder().setHost("localhost").setPort(port).build(), data, ZooDefs.Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT),
					Op.create(new ZookeeperPathBuilder().setHost("localhost").setPort(port).addPath("tmp").build(), data, ZooDefs.Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT)

					));
			
			String localhost = InetAddress.getLocalHost().getHostName();
			if(!localhost.equals(host)) {
				zk.multi(Arrays.asList(
						Op.create(new ZookeeperPathBuilder().setHost(localhost).setPort(port).build(), data, ZooDefs.Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT),
						Op.create(new ZookeeperPathBuilder().setHost(localhost).setPort(port).addPath("tmp").build(), data, ZooDefs.Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT)

						));
				log.info("Registered at " + localhost);
			}
			
		}catch(KeeperException.NodeExistsException e) {
			log.warn("Already exists..");
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		} catch (KeeperException e) {
			log.error("Error with node creation",e);
		} catch (UnknownHostException e) {
			log.error("Error with node creation",e);

		}


		try {
			String path = new ZookeeperPathBuilder().setHost(host).setPort(port).addPaths(Arrays.asList("tmp",id)).build();
			Stat stat = zk.exists(path, true);
			if(stat != null) {
				log.info("Path found " + path + " ...deleting");
				zk.delete(path, -1);
			}
			zk.create(path, data, ZooDefs.Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);

		}catch(InterruptedException e) {
			Thread.currentThread().interrupt();
			log.error("Interrupted registering of zookeeper data for id " + id,e);

		} catch (KeeperException e) {
			log.error("Error registering zookeeper data ",e);
			throw new RuntimeException(e);
		}
	}

	public void close() {
		try {
			zk.close();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
	}

	@Override
	public void process(WatchedEvent event) {
		if(event.getState() == KeeperState.Expired) {
			zk = new ZookeeperBuilder().setHost(host).setPort(port).setWatcher(this).build();

		}		
	}

}
