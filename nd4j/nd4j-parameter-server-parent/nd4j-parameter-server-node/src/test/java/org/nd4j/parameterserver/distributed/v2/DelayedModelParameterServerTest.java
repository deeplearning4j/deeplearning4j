/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.parameterserver.distributed.v2;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.v2.transport.impl.DelayedDummyTransport;
import org.nd4j.parameterserver.distributed.v2.transport.impl.DummyTransport;

import java.util.ArrayList;

import static org.junit.Assert.assertEquals;

@Slf4j
public class DelayedModelParameterServerTest {
    private static final String rootId = "ROOT_NODE";

    @Test(timeout = 20000L)
    public void testBasicInitialization_1() throws Exception {
        val connector = new DummyTransport.Connector();
        val rootTransport = new DelayedDummyTransport(rootId, connector);

        connector.register(rootTransport);

        val rootServer = new ModelParameterServer(rootTransport, true);
        rootServer.launch();

        assertEquals(rootId, rootTransport.getUpstreamId());

        rootServer.shutdown();
    }

    @Test(timeout = 40000L)
    public void testBasicInitialization_2() throws Exception {
        for (int e = 0; e < 100; e++) {
            val connector = new DummyTransport.Connector();
            val rootTransport = new DelayedDummyTransport(rootId, connector);
            val clientTransportA = new DelayedDummyTransport("123", connector, rootId);
            val clientTransportB = new DelayedDummyTransport("1234", connector, rootId);

            connector.register(rootTransport, clientTransportA, clientTransportB);

            val rootServer = new ModelParameterServer(rootTransport, true);
            val clientServerA = new ModelParameterServer(clientTransportA, false);
            val clientServerB = new ModelParameterServer(clientTransportB, false);
            rootServer.launch();
            clientServerA.launch();
            clientServerB.launch();

            // since clientB starts AFTER clientA, we have to wait till MeshUpdate message is propagated, since ithis message is NOT blocking
            Thread.sleep(25);

            val meshR = rootTransport.getMesh();
            val meshA = clientTransportA.getMesh();
            val meshB = clientTransportB.getMesh();

            assertEquals("Root node failed",3, meshR.totalNodes());
            assertEquals("B node failed", 3, meshB.totalNodes());
            assertEquals("A node failed", 3, meshA.totalNodes());
            assertEquals(meshR, meshA);
            assertEquals(meshA, meshB);

            log.info("Iteration [{}] finished", e);
        }
    }

    @Test
    public void testUpdatesPropagation_1() throws Exception {
        val array = Nd4j.ones(10, 10);

        val connector = new DummyTransport.Connector();
        val rootTransport = new DelayedDummyTransport(rootId, connector);
        val clientTransportA = new DelayedDummyTransport("412334", connector, rootId);
        val clientTransportB = new DelayedDummyTransport("123441", connector, rootId);

        connector.register(rootTransport, clientTransportA, clientTransportB);

        val rootServer = new ModelParameterServer(rootTransport, true);
        val clientServerA = new ModelParameterServer(clientTransportA, false);
        val clientServerB = new ModelParameterServer(clientTransportB, false);
        rootServer.launch();
        clientServerA.launch();
        clientServerB.launch();

        val servers = new ArrayList<ModelParameterServer>();
        val transports = new ArrayList<DelayedDummyTransport>();
        for (int e = 0; e < 256; e++) {
            val clientTransport = new DelayedDummyTransport(String.valueOf(e), connector, rootId);
            val clientServer = new ModelParameterServer(clientTransport, false);

            connector.register(clientTransport);
            servers.add(clientServer);
            transports.add(clientTransport);

            clientServer.launch();

            log.info("Server [{}] started...", e);
        }

        // 259 == 256 + A+B+R
        assertEquals(259, rootTransport.getMesh().totalNodes());

        clientServerA.sendUpdate(array);

        val updatesR = rootServer.getUpdates();
        val updatesA = clientServerA.getUpdates();
        val updatesB = clientServerB.getUpdates();

        assertEquals(1, updatesR.size());
        assertEquals(1, updatesB.size());

        // we should NOT get this message back to A
        assertEquals(0, updatesA.size());
    }
}
