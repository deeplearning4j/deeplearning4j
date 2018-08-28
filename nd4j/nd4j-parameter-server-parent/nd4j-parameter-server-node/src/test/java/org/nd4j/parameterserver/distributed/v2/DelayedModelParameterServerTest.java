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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.AtomicBoolean;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.v2.enums.MeshBuildMode;
import org.nd4j.parameterserver.distributed.v2.transport.impl.DelayedDummyTransport;
import org.nd4j.parameterserver.distributed.v2.transport.impl.DummyTransport;
import org.nd4j.parameterserver.distributed.v2.util.AbstractSubscriber;

import java.util.ArrayList;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

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
        Thread.sleep(150);

        // 259 == 256 + A+B+R
        assertEquals(servers.size() + 3, rootTransport.getMesh().totalNodes());

        clientServerA.sendUpdate(array);

        Thread.sleep(150);

        val updatesR = rootServer.getUpdates();
        val updatesA = clientServerA.getUpdates();
        val updatesB = clientServerB.getUpdates();

        assertEquals(1, updatesR.size());
        assertEquals(1, updatesB.size());

        // we should NOT get this message back to A
        assertEquals(0, updatesA.size());

        for (int e = 0; e < servers.size(); e++) {
            val s = servers.get(e);
            assertEquals("Failed at node [" + e + "]", 1, s.getUpdates().size());
        }
    }

    @Test
    public void testModelAndUpdaterParamsUpdate_1() throws Exception {
        val config = VoidConfiguration.builder().meshBuildMode(MeshBuildMode.MESH).build();
        val connector = new DummyTransport.Connector();
        val rootTransport = new DelayedDummyTransport(rootId, connector, rootId, config);

        val updatedModel = new AtomicBoolean(false);
        val updatedUpdater = new AtomicBoolean(false);
        val gotGradients = new AtomicBoolean(false);

        connector.register(rootTransport);

        val servers = new ArrayList<ModelParameterServer>();
        val transports = new ArrayList<DummyTransport>();
        for (int e = 0; e < 128; e++) {
            val clientTransport = new DummyTransport(java.util.UUID.randomUUID().toString(), connector, rootId, config);
            val clientServer = new ModelParameterServer(config, clientTransport, false);

            servers.add(clientServer);
            transports.add(clientTransport);

            connector.register(clientTransport);

            clientServer.launch();
            log.info("Client [{}] started", e );
        }

        Thread.sleep(100);
        val rootMesh = rootTransport.getMesh();

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

        clientServer.addUpdaterParamsSubscriber(new AbstractSubscriber<INDArray>() {
            @Override
            public void onNext(INDArray array) {
                assertNotNull(array);
                updatedUpdater.set(true);
            }
        });

        clientServer.addModelParamsSubscriber(new AbstractSubscriber<INDArray>() {
            @Override
            public void onNext(INDArray array) {
                assertNotNull(array);
                updatedModel.set(true);
            }
        });

        clientServer.addUpdatesSubscriber(new AbstractSubscriber<INDArray>() {
            @Override
            public void onNext(INDArray array) {
                assertNotNull(array);
                assertEquals(Nd4j.linspace(1, 10, 100).reshape(10, 10), array);
                gotGradients.set(true);
            }
        });

        connector.register(clientTransport);

        clientServer.launch();

        Thread.sleep(100);

        // getting any server
        val serv = servers.get(96);
        serv.sendUpdate(Nd4j.linspace(1, 10, 100).reshape(10, 10));

        Thread.sleep(500);

        assertTrue(updatedModel.get());
        assertTrue(updatedUpdater.get());
        assertTrue(gotGradients.get());
    }
}
