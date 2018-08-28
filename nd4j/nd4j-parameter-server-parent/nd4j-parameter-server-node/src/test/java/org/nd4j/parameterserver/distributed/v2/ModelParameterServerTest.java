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
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.v2.enums.MeshBuildMode;
import org.nd4j.parameterserver.distributed.v2.transport.impl.DummyTransport;

import java.util.ArrayList;

import static org.junit.Assert.*;

@Slf4j
public class ModelParameterServerTest {
    private static final String rootId = "ROOT_NODE";

    @Test(timeout = 20000L)
    public void testBasicInitialization_1() throws Exception {
        val connector = new DummyTransport.Connector();
        val rootTransport = new DummyTransport(rootId, connector);

        connector.register(rootTransport);

        val rootServer = new ModelParameterServer(rootTransport, true);
        rootServer.launch();

        assertEquals(rootId, rootTransport.getUpstreamId());

        rootServer.shutdown();
    }

    @Test(timeout = 20000L)
    public void testBasicInitialization_2() throws Exception {
        val connector = new DummyTransport.Connector();
        val rootTransport = new DummyTransport(rootId, connector);
        val clientTransportA = new DummyTransport("123", connector, rootId);
        val clientTransportB = new DummyTransport("1234", connector, rootId);

        connector.register(rootTransport, clientTransportA, clientTransportB);

        val rootServer = new ModelParameterServer(rootTransport, true);
        val clientServerA = new ModelParameterServer(clientTransportA, false);
        val clientServerB = new ModelParameterServer(clientTransportB, false);
        rootServer.launch();
        clientServerA.launch();
        clientServerB.launch();

        val meshR = rootTransport.getMesh();
        val meshA = clientTransportA.getMesh();
        val meshB = clientTransportB.getMesh();

        assertEquals(3, meshA.totalNodes());
        assertEquals(meshR, meshA);
        assertEquals(meshA, meshB);
    }

    @Test
    public void testUpdatesPropagation_1() throws Exception {
        val connector = new DummyTransport.Connector();
        val rootTransport = new DummyTransport(rootId, connector);
        val clientTransportA = new DummyTransport("412334", connector, rootId);
        val clientTransportB = new DummyTransport("123441", connector, rootId);

        connector.register(rootTransport, clientTransportA, clientTransportB);

        val rootServer = new ModelParameterServer(rootTransport, true);
        val clientServerA = new ModelParameterServer(clientTransportA, false);
        val clientServerB = new ModelParameterServer(clientTransportB, false);
        rootServer.launch();
        clientServerA.launch();
        clientServerB.launch();

        val array = Nd4j.ones(10, 10);
        clientServerA.sendUpdate(array);

        val updatesR = rootServer.getUpdates();
        val updatesA = clientServerA.getUpdates();
        val updatesB = clientServerB.getUpdates();

        assertEquals(1, updatesR.size());
        assertEquals(1, updatesB.size());

        // we should NOT get this message back to A
        assertEquals(0, updatesA.size());
    }

    @Test //(timeout = 10000L)
    public void testReconnectPropagation_1() throws Exception {
        val config = VoidConfiguration.builder().meshBuildMode(MeshBuildMode.MESH).build();
        val connector = new DummyTransport.Connector();
        val rootTransport = new DummyTransport(rootId, connector, rootId, config);

        connector.register(rootTransport);

        val rootServer = new ModelParameterServer(config, rootTransport, true);
        rootServer.launch();

        val servers = new ArrayList<ModelParameterServer>();
        val transports = new ArrayList<DummyTransport>();
        for (int e = 0; e < 2048; e++) {
            val clientTransport = new DummyTransport(java.util.UUID.randomUUID().toString(), connector, rootId, config);
            val clientServer = new ModelParameterServer(config, clientTransport, false);

            servers.add(clientServer);
            transports.add(clientTransport);

            connector.register(clientTransport);

            clientServer.launch();
            //log.info("Client [{}] started", e );
        }

        // at this point we should have 2048 nodes within
        val rootMesh = rootTransport.getMesh();
        val originalVersion = rootMesh.getVersion();
        assertEquals(2048, rootMesh.getVersion());

        // all mesh structures should be equal
        for (val t:transports)
            assertEquals(rootMesh, t.getMesh());

        // now we're picking one server that'll play bad role
        val badServer = servers.get(23);
        val badTransport = transports.get(23);
        val badId = badTransport.id();
        val badNode = rootMesh.getNodeById(badId);

        val upstreamId = badNode.getUpstreamNode().getId();
        log.info("Upstream: [{}]; Number of downstreams: [{}]", upstreamId, badNode.numberOfDownstreams());

        connector.dropConnection(badId);
        val clientTransport = new DummyTransport(badId, connector, rootId);
        val clientServer = new ModelParameterServer(clientTransport, false);
        connector.register(clientTransport);

        clientServer.launch();

        // at this point we have re-registered node
        assertNotEquals(originalVersion, rootMesh.getVersion());
        val newNode = rootMesh.getNodeById(badId);
        val newUpstream = newNode.getUpstreamNode().getId();

        // after reconnect node should have 0 downstreams and new upstream
        assertNotEquals(upstreamId, newUpstream);
        assertEquals(0, newNode.numberOfDownstreams());
    }
}