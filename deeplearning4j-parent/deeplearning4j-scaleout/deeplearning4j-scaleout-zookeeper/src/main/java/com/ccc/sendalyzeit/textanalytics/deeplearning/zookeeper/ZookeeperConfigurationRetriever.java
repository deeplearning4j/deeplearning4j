package com.ccc.sendalyzeit.textanalytics.deeplearning.zookeeper;

import java.util.Arrays;
import java.util.List;

import org.apache.zookeeper.WatchedEvent;
import org.apache.zookeeper.Watcher;
import org.apache.zookeeper.Watcher.Event.KeeperState;
import org.apache.zookeeper.ZooKeeper;
import org.apache.zookeeper.data.Stat;

import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.Conf;
/**
 * Retrieves configuration data serialized by {@link com.ccc.sendalyzeit.textanalytics.mapreduce.hadoop.util.ZooKeeperConfigurationRegister}
 * @author Adam Gibson
 *
 */
public class ZookeeperConfigurationRetriever implements Watcher {

	private ZooKeeper keeper;
	private String host;
	private int port;
	private String id;

	public ZookeeperConfigurationRetriever( String host,
			int port, String id) {
		super();
		this.keeper = new ZookeeperBuilder().setHost(host).setPort(port).build();
		this.host = host;
		this.port = port;
		this.id = id;
	}


	public ZookeeperConfigurationRetriever(ZooKeeper keeper,String id) {
		super();
		this.keeper = keeper;
		this.id = id;
	}


	public Conf retrieve(Conf conf) throws Exception {
		Conf ret = retreive();
		for( String key : conf.keySet()) {
			conf.put(key,ret.get(key));
		}
		return conf;
	}
	public Conf retreive() throws Exception {
		Conf conf = new Conf();
		String path = new ZookeeperPathBuilder().addPaths(Arrays.asList("tmp",id)).setHost(host).setPort(port).build();
		Stat stat = keeper.exists(path, false);
		if(stat==null) {
			List<String> list = keeper.getChildren( new ZookeeperPathBuilder().setHost(host).setPort(port).addPath("tmp").build(), false);
			throw new IllegalStateException("Nothing found for " + path + " possible children include " + list);
		}
		String data = new String( keeper.getData(path, false, stat ) );
		String[] split = data.split("\n");
		for(String s : split) {
			if(s.isEmpty() || s.charAt(0)=='#')
				continue;
			String[] split2 = s.split("=");
			if(split2.length > 1)
				conf.put(split2[0], split2[1]);

		}

		return conf;
	}

	public void close() {
		try {
			keeper.close();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}
	}


	@Override
	public void process(WatchedEvent event) {
		if(event.getState() == KeeperState.Expired) {
			keeper = new ZookeeperBuilder().setHost(host).setPort(port).setWatcher(this).build();

		}		
	}
}
