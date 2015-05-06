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
