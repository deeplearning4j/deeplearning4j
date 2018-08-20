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
import org.nd4j.parameterserver.distributed.v2.transport.impl.DelayedDummyTransport;
import org.nd4j.parameterserver.distributed.v2.transport.impl.DummyTransport;

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

    @Test(timeout = 20000L)
    public void testBasicInitialization_2() throws Exception {
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

        val meshR = rootTransport.getMesh();
        val meshA = clientTransportA.getMesh();
        val meshB = clientTransportB.getMesh();

        assertEquals(3, meshA.totalNodes());
        assertEquals(meshR, meshA);
        assertEquals(meshA, meshB);
    }
}
