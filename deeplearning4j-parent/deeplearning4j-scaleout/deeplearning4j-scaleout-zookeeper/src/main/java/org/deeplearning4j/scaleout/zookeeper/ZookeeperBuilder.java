package org.deeplearning4j.scaleout.zookeeper;

import java.io.IOException;
import java.util.concurrent.CountDownLatch;

import org.apache.zookeeper.WatchedEvent;
import org.apache.zookeeper.Watcher;
import org.apache.zookeeper.Watcher.Event.KeeperState;
import org.apache.zookeeper.ZooKeeper;
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
	private CountDownLatch latch = new CountDownLatch(1);
	public ZookeeperBuilder() {
		host = "localhost";
		port = 2181;
		timeout = 10000;
		
	}
	
	public ZooKeeper build() {
		try {
			keeper = new ZooKeeper(host + ":" + port,timeout,this);
			latch.await();
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
		if(event.getState() == KeeperState.SyncConnected)
			latch.countDown();
		else if(event.getState() == KeeperState.Disconnected) {
			keeper = build();
		}
		else if(event.getState() == KeeperState.Expired) {
			keeper = build();
		}
		if(watcher!=null)
			watcher.process(event);
	}

	
	
	
}
