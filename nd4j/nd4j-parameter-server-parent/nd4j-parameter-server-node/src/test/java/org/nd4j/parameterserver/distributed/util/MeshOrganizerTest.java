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

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.parameterserver.distributed.enums.MeshBuildMode;

import java.util.ArrayList;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Slf4j
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
    public void testDistanceFromRoot_1() {
        val rootNode = new MeshOrganizer.Node(true);

        val node0 = rootNode.addDownstreamNode(new MeshOrganizer.Node());
        val node1 = node0.addDownstreamNode(new MeshOrganizer.Node());

        assertEquals(2, node1.distanceFromRoot());

        val node2 = node1.addDownstreamNode(new MeshOrganizer.Node());

        assertEquals(3, node2.distanceFromRoot());
    }

    @Test
    public void testNextCandidate_1() {
        val rootNode = new MeshOrganizer.Node(true);

        val node0 = rootNode.addDownstreamNode(new MeshOrganizer.Node());
        val node1 = rootNode.addDownstreamNode(new MeshOrganizer.Node());
        val node2 = rootNode.addDownstreamNode(new MeshOrganizer.Node());

        val c1_0 = node1.getNextCandidate(null);
        assertEquals(node1, c1_0);

        val nn = c1_0.addDownstreamNode(new MeshOrganizer.Node());
    }

    @Test
    public void testPushDownstream_1() {
        val rootNode = new MeshOrganizer.Node(true);

        for (int e = 0; e < MeshOrganizer.MAX_DOWNSTREAMS  * MeshOrganizer.MAX_DEPTH * 2; e++ )
            rootNode.pushDownstreamNode(new MeshOrganizer.Node());

        assertEquals(2, rootNode.numberOfDownstreams());
        assertEquals(MeshOrganizer.MAX_DOWNSTREAMS * MeshOrganizer.MAX_DEPTH  * 2, rootNode.numberOfDescendants());
    }

    @Test
    public void testBasicMesh_1() {
        val mesh = new MeshOrganizer(MeshBuildMode.SYMMETRIC_MODE);

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



        // smoke test
        for (int e = 0; e < 8192; e++)
            mesh.addNode(java.util.UUID.randomUUID().toString());


        for (val v: mesh.flatNodes())
            assertTrue(v.numberOfDownstreams() <= MeshOrganizer.MAX_DOWNSTREAMS);
    }

    @Test
    public void testBasicMesh_2() {
        val mesh = new MeshOrganizer(MeshBuildMode.WIDTH_FIRST);

        val node1 = mesh.addNode("192.168.1.1");
        val node2 = mesh.addNode("192.168.2.1");
        val node3 = mesh.addNode("192.168.2.2");

        assertEquals(4, mesh.totalNodes());
        assertEquals(3, mesh.getRootNode().numberOfDownstreams());

        val node4 = mesh.addNode("192.168.4.1");
        val node5 = mesh.addNode("192.168.5.1");

        assertEquals(2, node1.numberOfDownstreams());

        val node6 = mesh.addNode("192.168.6.1");
        val node7 = mesh.addNode("192.168.7.1");

        assertEquals(3, node1.numberOfDownstreams());
        assertEquals(1, node2.numberOfDownstreams());

        // now we just drop in 20 nodes
        for (int e = 0; e < 20; e++)
            mesh.addNode(java.util.UUID.randomUUID().toString());


        for (val v: mesh.flatNodes())
            assertTrue(v.numberOfDownstreams() <= MeshOrganizer.MAX_DOWNSTREAMS);

        // smoke test
        for (int e = 0; e < 8192; e++)
            mesh.addNode(java.util.UUID.randomUUID().toString());

        // and now we'll make sure there's no nodes with number of downstreams > MAX_DOWNSTREAMS
        for (val v: mesh.flatNodes())
            assertTrue(v.numberOfDownstreams() <= MeshOrganizer.MAX_DOWNSTREAMS);
    }

    @Test
    public void testBasicMesh_3() {
        val mesh = new MeshOrganizer(MeshBuildMode.DEPTH_FIRST);

        val node1 = mesh.addNode("192.168.1.1");
        val node2 = mesh.addNode("192.168.2.1");
        val node3 = mesh.addNode("192.168.2.2");

        assertEquals(4, mesh.totalNodes());
        assertEquals(1, mesh.getRootNode().numberOfDownstreams());

        val node4 = mesh.addNode("192.168.2.3");
        val node5 = mesh.addNode("192.168.2.4");
        val node6 = mesh.addNode("192.168.2.5");

        assertEquals(1, node1.numberOfDownstreams());
        assertEquals(1, node4.numberOfDownstreams());
        assertEquals(1, node5.numberOfDownstreams());

        assertEquals(1, node2.numberOfDownstreams());
    }

    @Test
    public void testBasicMesh_4() {
        val mesh = new MeshOrganizer(MeshBuildMode.DEPTH_FIRST);

        // smoke test
        for (int e = 0; e < 8192; e++)
            mesh.addNode(java.util.UUID.randomUUID().toString());

        // 8192 nodes + root node
        assertEquals(8193, mesh.totalNodes());

        // and now we'll make sure there's no nodes with number of downstreams > MAX_DOWNSTREAMS
        for (val v: mesh.flatNodes()) {
            assertTrue(v.numberOfDownstreams() <= MeshOrganizer.MAX_DOWNSTREAMS);
            assertTrue(v.distanceFromRoot() <= MeshOrganizer.MAX_DEPTH);
        }

        // each of the root nodes should have limited number of descendants
        val rootDownstreams = new ArrayList<MeshOrganizer.Node>(mesh.getRootNode().getDownstreamNodes());
        for (val r: rootDownstreams)
            assertTrue(r.numberOfDescendants() <= MeshOrganizer.MAX_DOWNSTREAMS * MeshOrganizer.MAX_DEPTH);
    }


    @Test
    public void testRemap_1() {
        val mesh = new MeshOrganizer(MeshBuildMode.SYMMETRIC_MODE);

        val node1 = mesh.addNode("192.168.1.1");
        val node2 = mesh.addNode("192.168.2.1");
        val node3 = mesh.addNode("192.168.2.2");

        mesh.markNodeOffline(node3);

        assertEquals(3, mesh.getRootNode().numberOfDownstreams());

        val node4 = mesh.addNode("192.168.1.7");

        assertEquals(1, node1.numberOfDownstreams());


        mesh.remapNode(node4);

        assertEquals(1, node2.numberOfDownstreams());
        assertEquals(0, node1.numberOfDownstreams());
    }


    @Test
    public void testEquality_1() throws Exception {
        val node1 = MeshOrganizer.Node.builder()
                .ip("192.168.0.1")
                .port(38912)
                .build();

        val node2 = MeshOrganizer.Node.builder()
                .ip("192.168.0.1")
                .port(38912)
                .build();

        val node3 = MeshOrganizer.Node.builder()
                .ip("192.168.0.1")
                .port(38913)
                .build();

        val node4 = MeshOrganizer.Node.builder()
                .ip("192.168.0.2")
                .port(38912)
                .build();

        assertEquals(node1, node2);

        assertNotEquals(node1, node3);
        assertNotEquals(node1, node4);
        assertNotEquals(node3, node4);
    }


    @Test
    public void testEquality_2() throws Exception {
        val node1 = MeshOrganizer.Node.builder()
                .ip("192.168.0.1")
                .port(38912)
                .build();

        val node2 = MeshOrganizer.Node.builder()
                .ip("192.168.0.1")
                .port(38912)
                .build();

        val node3 = MeshOrganizer.Node.builder()
                .ip("192.168.0.1")
                .port(38912)
                .build();

        node1.addDownstreamNode(MeshOrganizer.Node.builder().ip("192.168.1.3").build()).addDownstreamNode(MeshOrganizer.Node.builder().ip("192.168.1.4").build()).addDownstreamNode(MeshOrganizer.Node.builder().ip("192.168.1.5").build());
        node2.addDownstreamNode(MeshOrganizer.Node.builder().ip("192.168.1.3").build()).addDownstreamNode(MeshOrganizer.Node.builder().ip("192.168.1.4").build()).addDownstreamNode(MeshOrganizer.Node.builder().ip("192.168.1.5").build());
        node3.addDownstreamNode(MeshOrganizer.Node.builder().ip("192.168.1.3").build()).addDownstreamNode(MeshOrganizer.Node.builder().ip("192.168.1.4").build());

        assertEquals(node1, node2);
        assertNotEquals(node1, node3);
    }


    @Test
    public void testEquality_3() throws Exception {
        val mesh1 = new MeshOrganizer();
        val mesh2 = new MeshOrganizer();
        val mesh3 = new MeshOrganizer(MeshBuildMode.DEPTH_FIRST);
        val mesh4 = new MeshOrganizer(MeshBuildMode.WIDTH_FIRST);

        assertEquals(mesh1, mesh2);
        assertNotEquals(mesh1, mesh3);
        assertNotEquals(mesh1, mesh4);
        assertNotEquals(mesh4, mesh3);
    }

    @Test
    public void testEquality_4() throws Exception {
        val mesh1 = new MeshOrganizer(MeshBuildMode.DEPTH_FIRST);
        val mesh2 = new MeshOrganizer(MeshBuildMode.DEPTH_FIRST);
        val mesh3 = new MeshOrganizer(MeshBuildMode.WIDTH_FIRST);

        mesh1.addNode("192.168.1.1");
        mesh2.addNode("192.168.1.1");
        mesh3.addNode("192.168.1.1");

        assertEquals(mesh1, mesh2);
        assertNotEquals(mesh1, mesh3);
    }
}