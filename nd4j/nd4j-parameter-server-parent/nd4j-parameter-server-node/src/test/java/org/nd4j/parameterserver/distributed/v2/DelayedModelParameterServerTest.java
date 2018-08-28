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

import io.reactivex.functions.Consumer;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.RandomUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.AtomicBoolean;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.v2.enums.MeshBuildMode;
import org.nd4j.parameterserver.distributed.v2.messages.impl.GradientsUpdateMessage;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.params.ModelParametersMessage;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.params.ModelParametersRequest;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.params.UpdaterParametersMessage;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.params.UpdaterParametersRequest;
import org.nd4j.parameterserver.distributed.v2.transport.MessageCallable;
import org.nd4j.parameterserver.distributed.v2.transport.impl.DelayedDummyTransport;
import org.nd4j.parameterserver.distributed.v2.transport.impl.DummyTransport;
import org.nd4j.parameterserver.distributed.v2.util.AbstractSubscriber;
import org.nd4j.parameterserver.distributed.v2.util.MessageSplitter;

import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

@Slf4j
public class DelayedModelParameterServerTest {
    private static final String rootId = "ROOT_NODE";

    @Before
    public void setUp() throws Exception {
        MessageSplitter.getInstance().reset();
    }

    @After
    public void setDown() throws Exception {
        MessageSplitter.getInstance().reset();
    }

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
        val conf = VoidConfiguration.builder().meshBuildMode(MeshBuildMode.PLAIN).build();
        val array = Nd4j.ones(10, 10);

        val connector = new DummyTransport.Connector();
        val rootTransport = new DelayedDummyTransport(rootId, connector, rootId, conf);
        val clientTransportA = new DelayedDummyTransport("412334", connector, rootId, conf);
        val clientTransportB = new DelayedDummyTransport("123441", connector, rootId, conf);

        connector.register(rootTransport, clientTransportA, clientTransportB);

        val rootServer = new ModelParameterServer(rootTransport, true);
        val clientServerA = new ModelParameterServer(clientTransportA, false);
        val clientServerB = new ModelParameterServer(clientTransportB, false);
        rootServer.launch();
        clientServerA.launch();
        clientServerB.launch();

        val servers = new ArrayList<ModelParameterServer>();
        val transports = new ArrayList<DelayedDummyTransport>();
        for (int e = 0; e < 128; e++) {
            val clientTransport = new DelayedDummyTransport(String.valueOf(e), connector, rootId, conf);
            val clientServer = new ModelParameterServer(clientTransport, false);

            connector.register(clientTransport);
            servers.add(clientServer);
            transports.add(clientTransport);

            clientServer.launch();

            //log.info("Server [{}] started...", e);
        }
        connector.blockUntilFinished();

        // 259 == 256 + A+B+R
        assertEquals(servers.size() + 3, rootTransport.getMesh().totalNodes());

        clientServerA.sendUpdate(array);

        connector.blockUntilFinished();

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
        val config = VoidConfiguration.builder().meshBuildMode(MeshBuildMode.PLAIN).build();
        val connector = new DummyTransport.Connector();
        val rootTransport = new DelayedDummyTransport(rootId, connector, rootId, config);
        rootTransport.addRequestConsumer(ModelParametersRequest.class, new Consumer<ModelParametersRequest>() {
            @Override
            public void accept(ModelParametersRequest modelParametersRequest) throws Exception {
                val msg = new ModelParametersMessage("123", Nd4j.create(10));
                msg.setRequestId(modelParametersRequest.getRequestId());
                rootTransport.sendMessage(msg, modelParametersRequest.getOriginatorId());
            }
        });

        rootTransport.addRequestConsumer(UpdaterParametersRequest.class, new Consumer<UpdaterParametersRequest>() {
            @Override
            public void accept(UpdaterParametersRequest updatersParametersRequest) throws Exception {
                val msg = new UpdaterParametersMessage("123", Nd4j.create(10));
                msg.setRequestId(updatersParametersRequest.getRequestId());
                rootTransport.sendMessage(msg, updatersParametersRequest.getOriginatorId());
            }
        });


        val updatedModel = new AtomicBoolean(false);
        val updatedUpdater = new AtomicBoolean(false);
        val gotGradients = new AtomicBoolean(false);

        connector.register(rootTransport);

        val counters = new AtomicInteger[128];
        val servers = new ArrayList<ModelParameterServer>();
        val transports = new ArrayList<DummyTransport>();
        for (int e = 0; e < 128; e++) {
            val clientTransport = new DelayedDummyTransport(java.util.UUID.randomUUID().toString(), connector, rootId, config);
            val clientServer = new ModelParameterServer(config, clientTransport, false);

            counters[e] = new AtomicInteger(0);

            val f = e;

            clientServer.addUpdatesSubscriber(new AbstractSubscriber<INDArray>() {
                @Override
                public void onNext(INDArray array) {
                    assertNotNull(array);
                    counters[f].incrementAndGet();
                }
            });

            servers.add(clientServer);
            transports.add(clientTransport);

            connector.register(clientTransport);

            clientServer.launch();
            //log.info("Client [{}] started", e );
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

        connector.blockUntilFinished();

        // getting any server
        val serv = servers.get(96);
        serv.sendUpdate(Nd4j.linspace(1, 10, 100).reshape(10, 10));

        connector.blockUntilFinished();

        for (int e = 0; e < 128; e++) {
            // we're skipping node 23 since it was reconnected, and has different MPS instance
            // and node 96, since it sends update
            if (e != 23 && e != 96)
                assertEquals("Failed at node: [" + e + "]", 1, counters[e].get());
        }

        assertTrue(updatedModel.get());
        assertTrue(updatedUpdater.get());
        assertTrue(gotGradients.get());
    }

    @Test
    public void testMeshConsistency_1() throws Exception {
        Nd4j.create(1);
        final int numMessages = 500;
        val rootCount = new AtomicInteger(0);
        val rootSum = new AtomicInteger(0);
        val counter = new AtomicInteger(0);
        val sum = new AtomicInteger(0);
        val config = VoidConfiguration.builder().meshBuildMode(MeshBuildMode.PLAIN).build();
        val connector = new DummyTransport.Connector();
        val rootTransport = new DelayedDummyTransport(rootId, connector, rootId, config);

        rootTransport.addPrecursor(GradientsUpdateMessage.class, new MessageCallable<GradientsUpdateMessage>() {
            @Override
            public void apply(GradientsUpdateMessage message) {
                val array = message.getPayload();
                rootSum.addAndGet(array.meanNumber().intValue());
                rootCount.incrementAndGet();
            }
        });
        connector.register(rootTransport);

        val counters = new AtomicInteger[16];
        val servers = new ArrayList<ModelParameterServer>();
        val transports = new ArrayList<DummyTransport>();
        for (int e = 0; e < 16; e++) {
            val clientTransport = new DelayedDummyTransport(java.util.UUID.randomUUID().toString(), connector, rootId, config);
            val clientServer = new ModelParameterServer(config, clientTransport, false);

            val f = e;
            counters[f] = new AtomicInteger(0);
            clientServer.addUpdatesSubscriber(new AbstractSubscriber<INDArray>() {
                @Override
                public void onNext(INDArray array) {
                    assertNotNull(array);
                    counters[f].incrementAndGet();
                }
            });

            servers.add(clientServer);
            transports.add(clientTransport);

            connector.register(clientTransport);

            clientServer.launch();
            //log.info("Client [{}] started", e );
        }


        val deductions = new int[servers.size()];
        for (int e = 0; e < numMessages; e++) {
            val f = RandomUtils.nextInt(0, servers.size());
            val server = servers.get(f);

            // later we'll reduce this number from expected number of updates
            deductions[f]++;

            server.sendUpdate(Nd4j.create(5).assign(e));
            sum.addAndGet(e);
        }

        connector.blockUntilFinished();

        // checking if master node got all updates we've sent
        assertEquals(numMessages, rootCount.get());
        assertEquals(sum.get(), rootSum.get());

        // now we're checking all nodes, they should get numMessages - messages that were sent through them
        for (int e = 0; e < servers.size(); e++) {
            val server = servers.get(e);
            assertEquals("Failed at node: [" + e + "]", numMessages - deductions[e], counters[e].get());
        }
    }
}
