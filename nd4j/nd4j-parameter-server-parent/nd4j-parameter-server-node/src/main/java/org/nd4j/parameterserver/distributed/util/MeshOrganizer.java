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

import lombok.*;
import lombok.extern.slf4j.Slf4j;


import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This class provides methods for ephemeral mesh network management
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class MeshOrganizer implements Serializable {
    private static final long serialVersionUID = 1L;

    // just shortcut to the root node of the tree
    private Node rootNode;

    // flattened map of the tree, ID -> Node
    private Map<String, Node> nodeMap = new HashMap<>();


    /**
     * This method adds new node to the network
     *
     * PLEASE NOTE: Default port 40123 is used
     * @param ip
     */
    protected Node  addNode(@NonNull String ip) {
        return addNode(ip, 40123);
    }

    /**
     * This methods adds new node to the network
     */
    public Node addNode(@NonNull String ip, @NonNull int port) {
        val node = Node.builder()
                .ip(ip)
                .port(port)
                .upstream(null)
                .build();

         return this.addNode(node);
    }


    public Node addNode(@NonNull Node node) {
        // if node isn't mapped yet - in this case we're mapping node automatically here
        if (node.getUpstreamNode() == null) {

        }

        // we should check if this node has any descendants
        if (node.numberOfDownstreams() > 0) {
            // if true - we should ensure they have their room in this mesh
        }

        // after all we add this node to the flattened map, for future access
        nodeMap.put(node.getId(), node);

        return node;
    }


    /**
     * This method removes  node from tree
     */
    public void removeNode() {
        //
    }


    /**
     * This method returns true, if node is known
     * @return
     */
    public boolean isKnownNode() {
        return false;
    }


    /**
     * This method reconnects given node to another node
     */
    public void remapNode() {
        //
    }

    /**
     * This method returns upstream connection for a given node
     */
    public void getUpstreamForNode() {
        //
    }

    /**
     * This method returns downstream connections for a given node
     */
    public void getDownstreamForNode() {
        //
    }

    /**
     * This method returns total number of nodes below given one
     * @return
     */
    public long numberOfDescendantsOfNode() {
        return rootNode.numberOfDescendants();
    }

    /**
     * This method returns total number of nodes in this mesh
     *
     * PLESE NOTE: this method INCLUDES root node
     * @return
     */
    public long totalNodes() {
        return rootNode.numberOfDescendants() + 1;
    }

    /**
     * This method returns size of flattened map of nodes.
     * Suited for tests.
     *
     * @return
     */
    protected long flatSize() {
        return (long) nodeMap.size();
    }

    /**
     * This method returns Node representing given Id
     * @return
     */
    protected Node getNodeById() {
        return null;
    }

    /**
     * This method returns Node representing given IP
     * @return
     */
    protected Node getNodeByIp() {
        return null;
    }

    /**
     * This class represents basic tree node
     */
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Data
    public static class Node implements Serializable, Comparable<Node> {
        private static final long serialVersionUID = 1L;

        @Getter private String id;
        @Getter private String ip;
        @Getter private int port;

        @Getter(AccessLevel.NONE) @Setter(AccessLevel.NONE) private Node upstream;

        @Getter(AccessLevel.NONE) @Setter(AccessLevel.NONE) @Builder.Default private Collection<Node> downstream = new ArrayList<>();

        /**
         * This method adds downstream node to the list of connections
         * @param node
         * @return
         */
        protected Node addDownstreamNode(@NonNull Node node) {
            this.downstream.add(node);
            return node;
        }

        /**
         * This method allows to set master node for this node
         * @param node
         * @return
         */
        protected Node setUpstreamNode(@NonNull Node node) {
            this.upstream = node;
            return node;
        }

        /**
         * This method returns the node this one it connected to
         * @return
         */
        protected Node getUpstreamNode() {
            return upstream;
        }

        /**
         * This method returns number of downstream nodes connected to this node
         * @return
         */
        public long numberOfDescendants() {
            val cnt = new AtomicLong(downstream.size());

            for (val n: downstream)
                cnt.addAndGet(n.numberOfDescendants());

            return cnt.get();
        }

        /**
         * This method returns number of nodes that has direct connection for this node
         * @return
         */
        public long numberOfDownstreams() {
            return downstream.size();
        }

        /**
         * This method returns collection of nodes that have direct connection to this node
         * @return
         */
        public Collection<Node> getDownstreamNodes() {
            return downstream;
        }

        @Override
        public int compareTo(@NonNull Node o) {
            return Long.compare(this.numberOfDescendants(), o.numberOfDescendants());
        }
    }
}
