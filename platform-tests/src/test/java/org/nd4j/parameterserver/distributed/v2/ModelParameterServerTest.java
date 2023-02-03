/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.parameterserver.distributed.v2;

import io.reactivex.functions.Consumer;
import lombok.extern.slf4j.Slf4j;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.v2.enums.MeshBuildMode;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.params.ModelParametersMessage;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.params.ModelParametersRequest;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.params.UpdaterParametersMessage;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.params.UpdaterParametersRequest;
import org.nd4j.parameterserver.distributed.v2.transport.UpdaterParametersProvider;
import org.nd4j.parameterserver.distributed.v2.transport.impl.DummyTransport;
import org.nd4j.parameterserver.distributed.v2.util.AbstractSubscriber;
import org.nd4j.parameterserver.distributed.v2.util.AbstractUpdatesHandler;
import org.nd4j.parameterserver.distributed.v2.util.MeshOrganizer;

import java.util.ArrayList;
import java.util.concurrent.LinkedTransferQueue;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@Disabled
@Tag(TagNames.FILE_IO)
@Tag(TagNames.DIST_SYSTEMS)
@NativeTag
public class ModelParameterServerTest extends BaseND4JTest {
    private static final String rootId = "ROOT_NODE";

    @Test()
    @Timeout(20000L)
    public void testBasicInitialization_1() throws Exception {
        var connector = new DummyTransport.Connector();
        var rootTransport = new DummyTransport(rootId, connector);

        connector.register(rootTransport);

        var rootServer = new ModelParameterServer(rootTransport, true);
        rootServer.launch();

        assertEquals(rootId, rootTransport.getUpstreamId());

        rootServer.shutdown();
    }

    @Test()
    @Timeout(20000L)
    public void testBasicInitialization_2() throws Exception {
        var connector = new DummyTransport.Connector();
        var rootTransport = new DummyTransport(rootId, connector);
        var clientTransportA = new DummyTransport("123", connector, rootId);
        var clientTransportB = new DummyTransport("1234", connector, rootId);

        connector.register(rootTransport, clientTransportA, clientTransportB);

        var rootServer = new ModelParameterServer(rootTransport, true);
        var clientServerA = new ModelParameterServer(clientTransportA, false);
        var clientServerB = new ModelParameterServer(clientTransportB, false);
        rootServer.launch();
        clientServerA.launch();
        clientServerB.launch();

        var meshR = rootTransport.getMesh();
        var meshA = clientTransportA.getMesh();
        var meshB = clientTransportB.getMesh();

        assertEquals(3, meshA.totalNodes());
        assertEquals(meshR, meshA);
        assertEquals(meshA, meshB);
    }

    @Test
    public void testUpdatesPropagation_1() throws Exception {
        var connector = new DummyTransport.Connector();
        var rootTransport = new DummyTransport(rootId, connector);
        var clientTransportA = new DummyTransport("412334", connector, rootId);
        var clientTransportB = new DummyTransport("123441", connector, rootId);

        connector.register(rootTransport, clientTransportA, clientTransportB);

        var rootServer = new ModelParameterServer(rootTransport, true);
        var clientServerA = new ModelParameterServer(clientTransportA, false);
        var clientServerB = new ModelParameterServer(clientTransportB, false);
        rootServer.launch();
        clientServerA.launch();
        clientServerB.launch();

        var array = Nd4j.ones(10, 10);
        clientServerA.sendUpdate(array);

        var updatesR = rootServer.getUpdates();
        var updatesA = clientServerA.getUpdates();
        var updatesB = clientServerB.getUpdates();

        assertEquals(1, updatesR.size());
        assertEquals(1, updatesB.size());

        // we should NOT get this message back to A
        assertEquals(0, updatesA.size());
    }

    @Test// (timeout = 30000L)
    public void testReconnectPropagation_1() throws Exception {
        var config = VoidConfiguration.builder().meshBuildMode(MeshBuildMode.MESH).build();
        var connector = new DummyTransport.Connector();
        var rootTransport = new DummyTransport(rootId, connector, rootId, config);

        connector.register(rootTransport);

        var rootServer = new ModelParameterServer(config, rootTransport, true);
        rootServer.addUpdatesSubscriber(new AbstractUpdatesHandler() {
            @Override
            public INDArray getParametersArray() {
                return Nd4j.create(10, 10);
            }

            @Override
            public void onNext(INDArray array) {

            }
        });
        rootServer.launch();

        var servers = new ArrayList<ModelParameterServer>();
        var transports = new ArrayList<DummyTransport>();
        for (int e = 0; e < 128; e++) {
            var clientTransport = new DummyTransport(java.util.UUID.randomUUID().toString(), connector, rootId, config);
            var clientServer = new ModelParameterServer(config, clientTransport, false);

            servers.add(clientServer);
            transports.add(clientTransport);

            connector.register(clientTransport);

            clientServer.launch();
            //log.info("Client [{}] started", e );
        }

        // at this point we should have 2048 nodes within
        var rootMesh = rootTransport.getMesh();
        var originalVersion = rootMesh.getVersion();
        assertEquals(128, rootMesh.getVersion());

        // all mesh structures should be equal
        for (var t:transports)
            assertEquals(rootMesh, t.getMesh());

        // now we're picking one server that'll play bad role
        var badServer = servers.get(23);
        var badTransport = transports.get(23);
        var badId = badTransport.id();
        var badNode = rootMesh.getNodeById(badId);

        var upstreamId = badNode.getUpstreamNode().getId();
        log.info("Upstream: [{}]; Number of downstreams: [{}]", upstreamId, badNode.numberOfDownstreams());

        connector.dropConnection(badId);
        var clientTransport = new DummyTransport(badId, connector, rootId);
        var clientServer = new ModelParameterServer(clientTransport, false);
        connector.register(clientTransport);

        clientServer.launch();

        // at this point we have re-registered node
        assertNotEquals(originalVersion, rootMesh.getVersion());
        var newNode = rootMesh.getNodeById(badId);
        var newUpstream = newNode.getUpstreamNode().getId();

        // after reconnect node should have 0 downstreams and new upstream
        assertNotEquals(upstreamId, newUpstream);
        assertEquals(0, newNode.numberOfDownstreams());
    }


    @Test
    public void testModelAndUpdaterParamsUpdate_1() throws Exception {
        var config = VoidConfiguration.builder().meshBuildMode(MeshBuildMode.PLAIN).build();
        var connector = new DummyTransport.Connector();
        var rootTransport = new DummyTransport(rootId, connector, rootId, config);
        rootTransport.addRequestConsumer(ModelParametersRequest.class, new Consumer<ModelParametersRequest>() {
            @Override
            public void accept(ModelParametersRequest modelParametersRequest) throws Exception {
                var msg = new ModelParametersMessage("123", Nd4j.create(10));
                msg.setRequestId(modelParametersRequest.getRequestId());
                rootTransport.sendMessage(msg, modelParametersRequest.getOriginatorId());
            }
        });

        rootTransport.addRequestConsumer(UpdaterParametersRequest.class, new Consumer<UpdaterParametersRequest>() {
            @Override
            public void accept(UpdaterParametersRequest updatersParametersRequest) throws Exception {
                var msg = new UpdaterParametersMessage("123", Nd4j.create(10));
                msg.setRequestId(updatersParametersRequest.getRequestId());
                rootTransport.sendMessage(msg, updatersParametersRequest.getOriginatorId());
            }
        });

        var updatedModel = new AtomicBoolean(false);
        var updatedUpdater = new AtomicBoolean(false);
        var gotGradients = new AtomicBoolean(false);

        connector.register(rootTransport);

        var servers = new ArrayList<ModelParameterServer>();
        var transports = new ArrayList<DummyTransport>();
        for (int e = 0; e < 128; e++) {
            var clientTransport = new DummyTransport(java.util.UUID.randomUUID().toString(), connector, rootId, config);
            var clientServer = new ModelParameterServer(config, clientTransport, false);

            servers.add(clientServer);
            transports.add(clientTransport);

            connector.register(clientTransport);

            clientServer.launch();
            log.info("Client [{}] started", e );
        }

        Thread.sleep(100);
        var rootMesh = rootTransport.getMesh();

        // now we're picking one server that'll play bad role
        var badServer = servers.get(23);
        var badTransport = transports.get(23);
        var badId = badTransport.id();
        var badNode = rootMesh.getNodeById(badId);

        var upstreamId = badNode.getUpstreamNode().getId();
        log.info("Upstream: [{}]; Number of downstreams: [{}]", upstreamId, badNode.numberOfDownstreams());

        connector.dropConnection(badId);
        var clientTransport = new DummyTransport(badId, connector, rootId);
        var clientServer = new ModelParameterServer(clientTransport, false);

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

        clientServer.addUpdatesSubscriber(new AbstractUpdatesHandler() {
            @Override
            public INDArray getParametersArray() {
                return null;
            }

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
        var serv = servers.get(96);
        serv.sendUpdate(Nd4j.linspace(1, 10, 100).reshape(10, 10));

        connector.blockUntilFinished();

        assertTrue(updatedModel.get());
        assertTrue(updatedUpdater.get());
        assertTrue(gotGradients.get());
    }

    @Test
    public void testModelAndUpdaterParamsUpdate_2() throws Exception {
        Nd4j.create(1);
        var config = VoidConfiguration.builder().meshBuildMode(MeshBuildMode.MESH).build();
        var connector = new DummyTransport.Connector();
        var rootTransport = new DummyTransport(rootId, connector, rootId, config);
        var rootServer = new ModelParameterServer(config, rootTransport, true);
        var rootUpdatesCounter = new AtomicInteger(0);
        rootTransport.addRequestConsumer(ModelParametersRequest.class, new Consumer<ModelParametersRequest>() {
            @Override
            public void accept(ModelParametersRequest modelParametersRequest) throws Exception {
                var msg = new ModelParametersMessage(java.util.UUID.randomUUID().toString(), Nd4j.create(10));
                msg.setRequestId(modelParametersRequest.getRequestId());
                rootTransport.sendMessage(msg, modelParametersRequest.getOriginatorId());
            }
        });

        rootTransport.addRequestConsumer(UpdaterParametersRequest.class, new Consumer<UpdaterParametersRequest>() {
            @Override
            public void accept(UpdaterParametersRequest updatersParametersRequest) throws Exception {
                var msg = new UpdaterParametersMessage(java.util.UUID.randomUUID().toString(), Nd4j.create(10));
                msg.setRequestId(updatersParametersRequest.getRequestId());
                rootTransport.sendMessage(msg, updatersParametersRequest.getOriginatorId());
            }
        });

        rootServer.addUpdatesSubscriber(new AbstractUpdatesHandler() {
            @Override
            public INDArray getParametersArray() {
                return Nd4j.create(10, 10);
            }

            @Override
            public void onNext(INDArray array) {
                assertNotNull(array);
                rootUpdatesCounter.incrementAndGet();
            }
        });
        connector.register(rootTransport);
        rootServer.launch();

        var updatedModel = new AtomicBoolean(false);
        var updatedUpdater = new AtomicBoolean(false);
        var gotGradients = new AtomicBoolean(false);


        var servers = new ArrayList<ModelParameterServer>();
        var transports = new ArrayList<DummyTransport>();
        for (int e = 0; e < 32; e++) {
            var clientTransport = new DummyTransport(String.valueOf(e), connector, rootId, config);
            var clientServer = new ModelParameterServer(config, clientTransport, false);

            clientServer.configure(config, clientTransport, new UpdaterParametersProvider() {
                @Override
                public INDArray getUpdaterParameters() {
                    return Nd4j.create(10, 10);
                }
            });
            servers.add(clientServer);
            transports.add(clientTransport);

            connector.register(clientTransport);

            clientServer.launch();
            log.info("Client [{}] started", e );
        }

        Thread.sleep(100);
        var rootMesh = rootTransport.getMesh();

        // now we're picking one server that'll play bad role
        var badServer = servers.get(23);
        var badTransport = transports.get(23);
        var badId = badTransport.id();
        var badNode = rootMesh.getNodeById(badId);

        var upstreamId = badNode.getUpstreamNode().getId();
        log.info("Upstream: [{}]; Number of downstreams: [{}]", upstreamId, badNode.numberOfDownstreams());

        connector.dropConnection(badId);
        var clientTransport = new DummyTransport(badId, connector, rootId);
        var clientServer = new ModelParameterServer(clientTransport, false);

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

        clientServer.addUpdatesSubscriber(new AbstractUpdatesHandler() {
            @Override
            public INDArray getParametersArray() {
                return null;
            }

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

        log.info("New upstream: {}", clientTransport.getMesh().getRootNode().getId());

        // getting any server
        var serv = servers.get(27);
        serv.sendUpdate(Nd4j.linspace(1, 10, 100).reshape(10, 10));

        connector.blockUntilFinished();

        int failedCnt = 0;
        for (int e = 0; e < 32; e++) {
            // we're skipping node 23 since it was reconnected, and has different MPS instance
            // and node 96, since it sends update
            if (e != 23 && e != 27)
                if (servers.get(e).getUpdates().size() == 0)
                    failedCnt++;
        }

        assertEquals(0, failedCnt,"Some nodes got no updates:");

        assertTrue(updatedModel.get());
        assertTrue(gotGradients.get());
        assertTrue(updatedUpdater.get());
    }


    @Test
    public void testLinearPropagation_1() throws Exception {
        Nd4j.create(1);
        var config = VoidConfiguration.builder().meshBuildMode(MeshBuildMode.MESH).build();
        var connector = new DummyTransport.Connector();
        var rootTransport = new DummyTransport(rootId, connector, rootId, config);
        var rootServer = new ModelParameterServer(config, rootTransport, true);
        var rootUpdatesCounter = new AtomicInteger(0);
        rootTransport.addRequestConsumer(ModelParametersRequest.class, new Consumer<ModelParametersRequest>() {
            @Override
            public void accept(ModelParametersRequest modelParametersRequest) throws Exception {
                var msg = new ModelParametersMessage("123", Nd4j.create(10));
                msg.setRequestId(modelParametersRequest.getRequestId());
                rootTransport.sendMessage(msg, modelParametersRequest.getOriginatorId());
            }
        });

        rootTransport.addRequestConsumer(UpdaterParametersRequest.class, new Consumer<UpdaterParametersRequest>() {
            @Override
            public void accept(UpdaterParametersRequest updatersParametersRequest) throws Exception {
                var msg = new UpdaterParametersMessage("123", Nd4j.create(10));
                msg.setRequestId(updatersParametersRequest.getRequestId());
                rootTransport.sendMessage(msg, updatersParametersRequest.getOriginatorId());
            }
        });

        rootServer.addUpdatesSubscriber(new AbstractUpdatesHandler() {
            @Override
            public INDArray getParametersArray() {
                return null;
            }

            @Override
            public void onNext(INDArray array) {
                assertNotNull(array);
                rootUpdatesCounter.incrementAndGet();
            }
        });
        connector.register(rootTransport);
        rootServer.launch();

        var servers = new ArrayList<ModelParameterServer>();
        var transports = new ArrayList<DummyTransport>();
        for (int e = 0; e < 7; e++) {
            var clientTransport = new DummyTransport(String.valueOf(e), connector, rootId, config);
            var clientServer = new ModelParameterServer(config, clientTransport, false);

            servers.add(clientServer);
            transports.add(clientTransport);

            connector.register(clientTransport);

            clientServer.launch();
            log.info("Client [{}] started", e );
        }

        var mesh = rootTransport.getMesh();
        var rootNode = mesh.getRootNode();
        var nodesForRemap = new LinkedTransferQueue<MeshOrganizer.Node>();
        MeshOrganizer.Node lastNode = null;
        int cnt = 0;
        for (var d: rootNode.getDownstreamNodes()) {
            assertEquals(0, d.numberOfDownstreams());
            assertEquals(0, d.numberOfDownstreams());
            if (cnt++ > 0) {
                rootNode.removeFromDownstreams(d);
                lastNode.addDownstreamNode(d);
                lastNode = d;
            } else
                lastNode = d;
        }
        assertEquals(1, rootNode.numberOfDownstreams());

        // now we want to ensure that all nodes have only 1 downstream, and last node has 0 downstreams
        var nodes = new ArrayList<MeshOrganizer.Node>(mesh.flatNodes());
        for (var n:nodes) {
            if (!n.getId().equals("6"))
                assertEquals(1, n.numberOfDownstreams());
            else
                assertEquals(0, n.numberOfDownstreams());
        }

        // update all mesh copies, just to be sure
        for (int e = 0; e < 7; e++) {
            var t = transports.get(e);
            t.setMesh(mesh);
        }

        var middleTransport = transports.get(3);
        log.info("Upstream ID: [{}]", middleTransport.getUpstreamId());


        var middleServer = servers.get(3);
        var update = Nd4j.create(10,10);
        middleServer.sendUpdate(update);
        connector.blockUntilFinished();

        // checking how many nodes got update
        int failCnt = 0;
        for (int e = 0; e < 7; e++) {
            var s = servers.get(e);
            if (e != 3)
                if (1 != s.getUpdates().size()) {
                    log.info("Node [{}] have no updates", e);
                    failCnt++;
                }
            else
                assertEquals(0, s.getUpdates().size());
        }
        assertEquals(0, failCnt);

        // now we're checking if root server got update
        assertEquals(1, rootUpdatesCounter.get());
    }
}