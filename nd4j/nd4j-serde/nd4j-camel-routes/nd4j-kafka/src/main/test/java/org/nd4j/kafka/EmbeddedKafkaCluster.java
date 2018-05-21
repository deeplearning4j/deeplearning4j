/*-*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.nd4j.kafka;

import kafka.admin.AdminUtils;
import kafka.server.KafkaConfig;
import kafka.server.KafkaServer;
import org.I0Itec.zkclient.ZkClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

public class EmbeddedKafkaCluster {
    private static final Logger LOG = LoggerFactory.getLogger(EmbeddedKafkaCluster.class);

    private final List<Integer> ports;
    private final String zkConnection;
    private final Properties baseProperties;

    private final String brokerList;

    private final List<KafkaServer> brokers;
    private final List<File> logDirs;

    public EmbeddedKafkaCluster(String zkConnection) {
        this(zkConnection, new Properties());
    }

    public EmbeddedKafkaCluster(String zkConnection, Properties baseProperties) {
        this(zkConnection, baseProperties, Collections.singletonList(-1));
    }

    public EmbeddedKafkaCluster(String zkConnection, Properties baseProperties, List<Integer> ports) {
        this.zkConnection = zkConnection;
        this.ports = resolvePorts(ports);
        this.baseProperties = baseProperties;
        this.brokers = new ArrayList<KafkaServer>();
        this.logDirs = new ArrayList<File>();

        this.brokerList = constructBrokerList(this.ports);
    }

    public ZkClient getZkClient() {
        for (KafkaServer server : brokers) {
            return server.zkClient();
        }
        return null;
    }

    public void createTopics(String... topics) {
        for (String topic : topics) {
            AdminUtils.createTopic(getZkClient(), topic, 2, 1, new Properties());
        }
    }

    private List<Integer> resolvePorts(List<Integer> ports) {
        List<Integer> resolvedPorts = new ArrayList<Integer>();
        for (Integer port : ports) {
            resolvedPorts.add(resolvePort(port));
        }
        return resolvedPorts;
    }

    private int resolvePort(int port) {
        if (port == -1) {
            return TestUtils.getAvailablePort();
        }
        return port;
    }

    private String constructBrokerList(List<Integer> ports) {
        StringBuilder sb = new StringBuilder();
        for (Integer port : ports) {
            if (sb.length() > 0) {
                sb.append(",");
            }
            sb.append("localhost:").append(port);
        }
        return sb.toString();
    }

    public void startup() {
        for (int i = 0; i < ports.size(); i++) {
            Integer port = ports.get(i);
            File logDir = TestUtils.constructTempDir("kafka-local");

            Properties properties = new Properties();
            properties.putAll(baseProperties);
            properties.setProperty("zookeeper.connect", zkConnection);
            properties.setProperty("broker.id", String.valueOf(i + 1));
            properties.setProperty("host.opName", "localhost");
            properties.setProperty("port", Integer.toString(port));
            properties.setProperty("log.dir", logDir.getAbsolutePath());
            properties.setProperty("num.partitions", String.valueOf(1));
            properties.setProperty("auto.create.topics.enable", String.valueOf(Boolean.TRUE));
            properties.setProperty("log.flush.interval.messages", String.valueOf(1));
            LOG.info("EmbeddedKafkaCluster: local directory: " + logDir.getAbsolutePath());

            KafkaServer broker = startBroker(properties);

            brokers.add(broker);
            logDirs.add(logDir);
        }
    }


    private KafkaServer startBroker(Properties props) {
        KafkaServer server = new KafkaServer(new KafkaConfig(props), new SystemTime());
        server.startup();
        return server;
    }

    public Properties getProps() {
        Properties props = new Properties();
        props.putAll(baseProperties);
        props.put("metadata.broker.list", brokerList);
        props.put("zookeeper.connect", zkConnection);
        return props;
    }

    public String getBrokerList() {
        return brokerList;
    }

    public List<Integer> getPorts() {
        return ports;
    }

    public String getZkConnection() {
        return zkConnection;
    }

    public void shutdown() {
        for (KafkaServer broker : brokers) {
            try {
                broker.shutdown();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        for (File logDir : logDirs) {
            try {
                TestUtils.deleteFile(logDir);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("EmbeddedKafkaCluster{");
        sb.append("brokerList='").append(brokerList).append('\'');
        sb.append('}');
        return sb.toString();
    }
}
