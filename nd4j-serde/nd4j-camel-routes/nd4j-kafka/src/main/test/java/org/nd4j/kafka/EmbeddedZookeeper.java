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

import org.apache.zookeeper.server.ServerCnxnFactory;
import org.apache.zookeeper.server.ZooKeeperServer;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.InetSocketAddress;

public class EmbeddedZookeeper {
    private int port = -1;
    private int tickTime = 500;

    private ServerCnxnFactory factory;
    private File snapshotDir;
    private File logDir;

    public EmbeddedZookeeper() {
        this(-1);
    }

    public EmbeddedZookeeper(int port) {
        this(port, 500);
    }

    public EmbeddedZookeeper(int port, int tickTime) {
        this.port = resolvePort(port);
        this.tickTime = tickTime;
    }

    private int resolvePort(int port) {
        if (port == -1) {
            return TestUtils.getAvailablePort();
        }
        return port;
    }

    public void startup() throws IOException {
        if (this.port == -1) {
            this.port = TestUtils.getAvailablePort();
        }
        this.factory = ServerCnxnFactory.createFactory(new InetSocketAddress("localhost", port), 1024);
        this.snapshotDir = TestUtils.constructTempDir("embeeded-zk/snapshot");
        this.logDir = TestUtils.constructTempDir("embeeded-zk/log");

        try {
            factory.startup(new ZooKeeperServer(snapshotDir, logDir, tickTime));
        } catch (InterruptedException e) {
            throw new IOException(e);
        }
    }


    public void shutdown() {
        factory.shutdown();
        try {
            TestUtils.deleteFile(snapshotDir);
        } catch (FileNotFoundException e) {
            // ignore
        }
        try {
            TestUtils.deleteFile(logDir);
        } catch (FileNotFoundException e) {
            // ignore
        }
    }

    public String getConnection() {
        return "localhost:" + port;
    }

    public void setPort(int port) {
        this.port = port;
    }

    public void setTickTime(int tickTime) {
        this.tickTime = tickTime;
    }

    public int getPort() {
        return port;
    }

    public int getTickTime() {
        return tickTime;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("EmbeddedZookeeper{");
        sb.append("connection=").append(getConnection());
        sb.append('}');
        return sb.toString();
    }
}
