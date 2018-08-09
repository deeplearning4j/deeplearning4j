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

package org.nd4j.parameterserver.distributed.util;

import lombok.val;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class MeshOrganizerTest {

    @Test(timeout = 1000L)
    public void testDescendantsCount_1() {
        val node = MeshOrganizer.Node.builder().build();

        val eNode = MeshOrganizer.Node.builder().build();
        eNode.addDownstreamNode(MeshOrganizer.Node.builder().build());

        node.addDownstreamNode(MeshOrganizer.Node.builder().build());
        node.addDownstreamNode(eNode);
        node.addDownstreamNode(MeshOrganizer.Node.builder().build());

        assertEquals(4, node.numberOfDescendants());
        assertEquals(3, node.numberOfDownstreams());
        assertEquals(1, eNode.numberOfDownstreams());
        assertEquals(1, eNode.numberOfDescendants());
    }

    @Test
    public void testBasicMesh_1() {
        val mesh = new MeshOrganizer();

        val node1 = mesh.addNode("192.168.1.1");
        val node2 = mesh.addNode("192.168.2.1");
        val node3 = mesh.addNode("192.168.2.2");

        assertEquals(4, mesh.totalNodes());
        assertEquals(3, mesh.getRootNode().numberOfDownstreams());

        // now we're adding one more node, and it should go elsewhere
        val node4 = mesh.addNode("192.168.3.1");

        assertEquals(5, mesh.totalNodes());
        assertEquals(3, mesh.getRootNode().numberOfDownstreams());

        // now we're adding one more node, and it should go elsewhere
        val node5 = mesh.addNode("192.168.4.1");
        val node6 = mesh.addNode("192.168.5.1");

        assertEquals(7, mesh.totalNodes());
        // now we expect flat distribution of descendants for all 3 first nodes
        assertEquals(1, node1.numberOfDownstreams());
        assertEquals(1, node2.numberOfDownstreams());
        assertEquals(1, node3.numberOfDownstreams());
    }
}