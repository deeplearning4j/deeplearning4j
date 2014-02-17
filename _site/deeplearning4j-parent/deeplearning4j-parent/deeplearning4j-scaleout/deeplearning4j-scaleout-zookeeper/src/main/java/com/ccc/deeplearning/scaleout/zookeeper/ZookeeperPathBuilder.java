package com.ccc.deeplearning.scaleout.zookeeper;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class ZookeeperPathBuilder {

	private String host;
	private int port;
	private List<String> paths;

	public ZookeeperPathBuilder() {
		paths = new ArrayList<String>();
	}

	public ZookeeperPathBuilder setPort(int port) {
		this.port = port;
		return this;
	}


	public ZookeeperPathBuilder addPaths(Collection<String> paths) {
		for(String s : paths)
			this.paths.add(s);
		return this;
	}
	public ZookeeperPathBuilder addPaths(String[] paths) {
		for(String s : paths)
			this.paths.add(s);
		return this;
	}

	public ZookeeperPathBuilder addPath(String path) {
		paths.add(path);
		return this;
	}

	public ZookeeperPathBuilder setHost(String host) {
		this.host = host;
		return this;
	}

	public String build() {
		StringBuffer sb = new StringBuffer();
		sb.append("/" + host + ":" + port);
		for(String s : paths) {
			sb.append("/" + s);
		}
		return sb.toString();
	}


}
