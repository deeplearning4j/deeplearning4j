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

package org.nd4j.parameterserver.distributed.v2.transport.impl;

import lombok.extern.slf4j.Slf4j;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.v2.enums.PropagationMode;
import org.nd4j.parameterserver.distributed.v2.messages.impl.GradientsUpdateMessage;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.handshake.HandshakeRequest;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.handshake.HandshakeResponse;
import org.nd4j.parameterserver.distributed.v2.transport.MessageCallable;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@Tag(TagNames.FILE_IO)
@Tag(TagNames.DIST_SYSTEMS)
@NativeTag
public class DummyTransportTest extends BaseND4JTest {

    @Test
    public void testBasicConnection_1() throws Exception {
        var counter = new AtomicInteger(0);
        var connector = new DummyTransport.Connector();
        var transportA = new DummyTransport("alpha", connector);
        var transportB = new DummyTransport("beta", connector);

        connector.register(transportA, transportB);
        transportB.addInterceptor(HandshakeRequest.class, message -> {
            // just increment
            counter.incrementAndGet();
        });

        transportA.sendMessage(new HandshakeRequest(), "beta");

        // we expect that message was delivered, and connector works
        assertEquals(1, counter.get());
    }

    @Test
    public void testHandshake_1() throws Exception {
        var counter = new AtomicInteger(0);
        var connector = new DummyTransport.Connector();
        var transportA = new DummyTransport("alpha", connector);
        var transportB = new DummyTransport("beta", connector);

        connector.register(transportA, transportB);
        transportB.addInterceptor(HandshakeResponse.class, (HandshakeResponse message) -> {
            // just increment
            assertNotNull(message);

            assertNotNull(message.getMesh());

            counter.incrementAndGet();
        });

        transportB.sendMessage(new HandshakeRequest(), "alpha");

        // we expect that message was delivered, and connector works
        assertEquals(1, counter.get());
    }

    @Test
    public void testMeshPropagation_1() throws Exception {
        var counter = new AtomicInteger(0);
        var connector = new DummyTransport.Connector();
        var transportA = new DummyTransport("alpha", connector);
        var transportB = new DummyTransport("beta", connector);
        var transportG = new DummyTransport("gamma", connector);
        var transportD = new DummyTransport("delta", connector);

        connector.register(transportA, transportB, transportG, transportD);


        transportB.sendMessage(new HandshakeRequest(), "alpha");
        transportG.sendMessage(new HandshakeRequest(), "alpha");
        transportD.sendMessage(new HandshakeRequest(), "alpha");

        var meshA = transportA.getMesh();
        var meshB = transportB.getMesh();
        var meshG = transportG.getMesh();
        var meshD = transportD.getMesh();

        // versions should be equal
        assertEquals(meshA.getVersion(), meshB.getVersion());
        assertEquals(meshA.getVersion(), meshG.getVersion());
        assertEquals(meshA.getVersion(), meshD.getVersion());

        // and meshs in general too
        assertEquals(meshA, meshB);
        assertEquals(meshA, meshG);
        assertEquals(meshA, meshD);

        assertTrue(meshA.isKnownNode("alpha"));
        assertTrue(meshA.isKnownNode("beta"));
        assertTrue(meshA.isKnownNode("gamma"));
        assertTrue(meshA.isKnownNode("delta"));

        var node = meshB.getNodeById("alpha");
        assertTrue(node.isRootNode());
    }

    @Test
    public void testUpdatesPropagation_1() throws Exception {
        var counter = new AtomicInteger(0);
        var connector = new DummyTransport.Connector();
        var transportA = new DummyTransport("alpha", connector);
        var transportB = new DummyTransport("beta", connector);
        var transportG = new DummyTransport("gamma", connector);
        var transportD = new DummyTransport("delta", connector);

        connector.register(transportA, transportB, transportG, transportD);

        transportB.sendMessage(new HandshakeRequest(), "alpha");
        transportG.sendMessage(new HandshakeRequest(), "alpha");
        transportD.sendMessage(new HandshakeRequest(), "alpha");


        var f = new MessageCallable<GradientsUpdateMessage>() {
            @Override
            public void apply(GradientsUpdateMessage message) {
                var update = message.getPayload();
                counter.addAndGet(update.sumNumber().intValue());
            }
        };

        transportA.addPrecursor(GradientsUpdateMessage.class, f);
        transportB.addPrecursor(GradientsUpdateMessage.class, f);
        transportG.addPrecursor(GradientsUpdateMessage.class, f);
        transportD.addPrecursor(GradientsUpdateMessage.class, f);

        var array = Nd4j.ones(10, 10);

        var msg = new GradientsUpdateMessage("message", array);
        msg.setOriginatorId("beta");
        transportB.propagateMessage(msg, PropagationMode.BOTH_WAYS);

        // we expect that each of the nodes gets this message
        assertEquals(400, counter.get());
    }

    @Test
    public void testReconnectAfterFailure_1() throws Exception {
        var counter = new AtomicInteger(0);
        var connector = new DummyTransport.Connector();
        var transportA = new DummyTransport("alpha", connector);
        var transportB = new DummyTransport("beta", connector);
        var transportG = new DummyTransport("gamma", connector);
        var transportD = new DummyTransport("delta", connector);
        var transportE = new DummyTransport("epsilon", connector);
        var transportZ = new DummyTransport("zeta", connector);
        var transportT = new DummyTransport("theta", connector);

        connector.register(transportA, transportB, transportG, transportD, transportE, transportZ, transportT);

        transportB.sendMessage(new HandshakeRequest(), "alpha");
        transportG.sendMessage(new HandshakeRequest(), "alpha");
        transportD.sendMessage(new HandshakeRequest(), "alpha");
        transportE.sendMessage(new HandshakeRequest(), "alpha");
        transportZ.sendMessage(new HandshakeRequest(), "alpha");
        transportT.sendMessage(new HandshakeRequest(), "alpha");

        var originalMeshA = transportA.getMesh();
        var originalMeshZ = transportZ.getMesh();

        assertEquals(originalMeshA, originalMeshZ);

        var version = originalMeshA.getVersion();
        var upstream = originalMeshZ.getUpstreamForNode("zeta");


        var restarted = new AtomicBoolean(false);
        var f = new MessageCallable<HandshakeResponse>() {
            @Override
            public void apply(HandshakeResponse message) {
                assertTrue(message.isRestart());
                restarted.set(true);
            }
        };
        transportZ.addPrecursor(HandshakeResponse.class, f);

        // this message basically says that Z is restarting
        transportZ.sendMessage(new HandshakeRequest(), "alpha");

        var newMesh = transportZ.getMesh();
        var newUpstream = newMesh.getUpstreamForNode("zeta");

        assertNotEquals(version, newMesh.getVersion());
        assertTrue(restarted.get());
    }
}