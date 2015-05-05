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

package org.deeplearning4j.hadoop.util;


import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.zookeeper.CreateMode;
import org.apache.zookeeper.KeeperException;
import org.apache.zookeeper.WatchedEvent;
import org.apache.zookeeper.Watcher;
import org.apache.zookeeper.Watcher.Event.KeeperState;
import org.apache.zookeeper.ZooDefs;
import org.apache.zookeeper.ZooKeeper;
import org.apache.zookeeper.data.Stat;
import org.deeplearning4j.scaleout.zookeeper.ZookeeperBuilder;
import org.deeplearning4j.scaleout.zookeeper.ZookeeperPathBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public  class HdfsLock implements Watcher {
	
	
	
	private ZooKeeper zk;
	private String host;
	private int port;
	private static final Logger log = LoggerFactory.getLogger(HdfsLock.class);
	public HdfsLock(String host,int port) {
		this.host = host;
		this.port = port;
		zk = new ZookeeperBuilder().setHost(host).setPort(port).build();

	}
	public HdfsLock(String host) {
		this(host,2181);

	}
	public boolean isLocked() throws KeeperException, InterruptedException {

		String lockPath = new ZookeeperPathBuilder().setHost(host).setPort(port).addPath("hdfslock2").build().replaceAll("//","/");
		Stat stat = zk.exists(lockPath, true);
		boolean ret = stat !=null;
		if(ret) {
			try {
				List<Path> paths = getPaths();
				Configuration conf = new Configuration();
				HdfsUtils.setHostForConf(conf);
				FileSystem system = FileSystem.get(conf);
				for(Path path : paths) {
					if(!system.exists(path)) {
						log.info("Paths found to be inconsistent. Auto clearing lock");
						ret = false;						
						zk.delete(lockPath, -1);
						break;
					}
				}

			} catch (Exception e) {
				log.error("Error accessing data, returning false",e);
				return false;
			}	
		}



		return ret;

	}
	
	
	public void create(Collection<Path> paths) throws Exception {
		String lockPath = new ZookeeperPathBuilder().setHost(host).setPort(port).addPath("hdfslock2").build().replaceAll("//","/");
		StringBuffer sb = new StringBuffer();
		for(Path path : paths) {
			sb.append(path.toUri().toString() + "\n");
		}
		try {
			zk.create(lockPath, sb.toString().getBytes(),ZooDefs.Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);
		}catch(KeeperException.SessionExpiredException e) {
			log.error("Session expired...trying again");
			if(zk !=null)
				zk.close();
			zk = new ZookeeperBuilder().setHost(host).setPort(port).build();
			create(paths);

		}
		catch(KeeperException.NodeExistsException e) {
			log.warn("Node exists...deleting");
			zk.delete(lockPath, -1);
			zk.create(lockPath, sb.toString().getBytes(),ZooDefs.Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);

		}
		catch(Exception e) {
			log.error("Unknown error...trying again");
			if(zk !=null)
				zk.close();
			zk = new ZookeeperBuilder().setHost(host).setPort(port).build();
			create(paths);

		}
	}
	public void delete() throws Exception {

		if(isLocked()) {
			String lockPath = new ZookeeperPathBuilder().setHost(host).setPort(port).addPath("hdfslock2").build().replaceAll("//","/");
			try {
				zk.delete(lockPath, -1);
			}catch(KeeperException.SessionExpiredException e) {
				log.error("Session expired...trying again");
				if(zk !=null)
					zk.close();
				zk = new ZookeeperBuilder().setHost(host).setPort(port).build();
				delete();

			}

		}
	}
	public void close() {
		try {
			zk.close();
		} catch (InterruptedException e) {
			log.info("Error closing lock",e);
			Thread.currentThread().interrupt();
		}
	}

	public List<Path> getPaths() throws Exception {
		List<Path> ret = new ArrayList<Path>();
		String lockPath = new ZookeeperPathBuilder().setHost(host).setPort(port).addPath("hdfslock2").build();
		Stat stat = zk.exists(lockPath, false);
		String data = new String( zk.getData(lockPath, false,stat) );
		String[] paths = data.split("\n");
		for(String s : paths) {
			ret.add(new Path(s));
		}
		return ret;
	}
	@Override
	public void process(WatchedEvent event) {
		if(event.getState() == KeeperState.Expired) {
			zk = new ZookeeperBuilder().setHost(host).setPort(port).setWatcher(this).build();

		}
	}

}