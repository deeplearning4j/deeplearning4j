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

import java.net.InetAddress;
import java.util.Arrays;
import java.util.List;

import org.apache.zookeeper.WatchedEvent;
import org.apache.zookeeper.Watcher;
import org.apache.zookeeper.Watcher.Event.KeeperState;
import org.apache.zookeeper.ZooKeeper;
import org.apache.zookeeper.data.Stat;
import org.canova.api.conf.Configuration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Retrieves configuration data serialized
 * by {@link ZooKeeperConfigurationRegister}
 * @author Adam Gibson
 *
 */
public class ZookeeperConfigurationRetriever implements Watcher {

    private ZooKeeper keeper;
    private String host;
    private int port;
    private String id;
    private static final Logger log = LoggerFactory.getLogger(ZookeeperConfigurationRetriever.class);



    public ZookeeperConfigurationRetriever( String host,
                                            int port, String id) {
        super();
        this.keeper = new ZookeeperBuilder().setHost(host).setPort(port).build();
        this.host = host;
        this.port = port;
        this.id = id;
    }



    public Configuration retrieve(String host) throws Exception {
        Configuration conf;
        String path = new ZookeeperPathBuilder().addPaths(Arrays.asList("tmp",id)).setHost(host).setPort(port).build();
        Stat stat = keeper.exists(path, false);
        if(stat == null) {
            List<String> list = keeper.getChildren( new ZookeeperPathBuilder().setHost(host).setPort(port).addPath("tmp").build(), false);
            throw new IllegalStateException("Nothing found for " + path + " possible children include " + list);
        }
        byte[] data = keeper.getData(path, false, stat ) ;
        conf = (Configuration) ZooKeeperConfigurationRegister.deserialize(data);


        return conf;
    }

    public Configuration retrieve() throws Exception {
        Configuration c = null;
        String localhost = InetAddress.getLocalHost().getHostName();

        String[] hosts = { host,"127.0.0.1","localhost",localhost };

        for(int i = 0; i < hosts.length; i++) {
            try {
                log.info("Attempting to retrieve conf from " + hosts[i]);
                c = retrieve(hosts[i]);
                if(c != null) {
                    log.info("Found from host " + hosts[i]);
                    break;
                }
            }catch(Exception e) {
                log.warn("Trying next host " + hosts[i] + " failed");
            }


        }

        log.info("Returning conf from host" + host);
        return c;
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
