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

package org.nd4j.parameterserver.distributed.v2.util;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.primitives.Atomic;
import org.nd4j.parameterserver.distributed.enums.MeshBuildMode;
import org.nd4j.parameterserver.distributed.enums.NodeStatus;


import java.io.Serializable;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This class provides methods for ephemeral mesh network management
 *
 * @author raver119@gmail.com
 */
@NoArgsConstructor
@Slf4j
public class MeshOrganizer implements Serializable {
    private static final long serialVersionUID = 1L;

    private MeshBuildMode buildMode = MeshBuildMode.SYMMETRIC_MODE;

    // this value determines max number of direct downstream connections for any given node (affects root node as well)
    public static final int MAX_DOWNSTREAMS = 3;

    // max distance from root
    public static final int MAX_DEPTH = 5;

    // just shortcut to the root node of the tree
    @Getter(AccessLevel.PUBLIC) private Node rootNode = new Node(true);

    // SortedSet, with sort by number of downstreams
    private List<Node> sortedNodes = new ArrayList<>();

    // flattened map of the tree, ID -> Node
    private Map<String, Node> nodeMap = new HashMap<>();

    // used in DEPTH_MODE
    private Node lastRoot = null;

    public MeshOrganizer(@NonNull MeshBuildMode mode) {
        this.buildMode = mode;
    }

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

    /**
     * This method adds new node to the mesh
     *
     * @param node
     * @return
     */
    public synchronized Node addNode(@NonNull Node node) {

        // if node isn't mapped yet - in this case we're mapping node automatically here
                switch (buildMode) {
                    case DEPTH_FIRST: {
                            sortedNodes.add(node);
                            return rootNode.pushDownstreamNode(node);
                        }
                    case WIDTH_FIRST: {
                            if (rootNode.numberOfDownstreams() < MAX_DOWNSTREAMS) {
                                if (lastRoot == null)
                                    lastRoot = node;

                                rootNode.addDownstreamNode(node);
                                sortedNodes.add(node);
                                return node;
                            }

                            sortedNodes.add(node);


                            // if lastRoot isn't full yet - we'll just add new node to it (this one)
                            if (lastRoot.numberOfDownstreams() < MAX_DOWNSTREAMS)
                                lastRoot.addDownstreamNode(node);
                            else { // or we'll pull next node otherwise
                                val upstream = lastRoot.getUpstreamNode();

                                Node c = upstream.getNextCandidate(lastRoot);

                                if (c == null)
                                    c = upstream.getNextCandidate(null);

                                // if we've maxed out number of downstreams - just step down
                                if (c.numberOfDownstreams() >= MAX_DOWNSTREAMS)
                                    c = c.downstream.get(0);

                                c.addDownstreamNode(node);
                                lastRoot = c;
                            }
                        };
                        break;
                    case SYMMETRIC_MODE: {
                            if (rootNode.numberOfDownstreams() < MAX_DOWNSTREAMS) {
                                if (lastRoot == null)
                                    lastRoot = node;

                                rootNode.addDownstreamNode(node);
                                sortedNodes.add(node);
                                return node;
                            }

                            val f = sortedNodes.get(0);
                            f.addDownstreamNode(node);

                            // we update sorted list, so we always know node with least number of
                            sortedNodes.add(node);
                            Collections.sort(sortedNodes);
                        }
                        break;
                    default:
                        throw new UnsupportedOperationException();
                }

        // after all we add this node to the flattened map, for future access
        nodeMap.put(node.getId(), node);

        return node;
    }

    /**
     * This method marks Node (specified by IP) as offline, and remaps its downstreams
     *
     * @param ip
     * @throws NoSuchElementException
     */
    public void markNodeOffline(@NonNull String ip) throws NoSuchElementException {
        markNodeOffline(getNodeByIp(ip));
    }

    /**
     * This method marks given Node as offline, remapping its downstreams
     * @param node
     */
    public  void markNodeOffline(@NonNull Node node) {
        synchronized (node) {
            node.status(NodeStatus.OFFLINE);

            for (val n : node.getDownstreamNodes())
                remapNode(n);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MeshOrganizer that = (MeshOrganizer) o;

        val bm = buildMode == that.buildMode;
        val rn = Objects.equals(rootNode, that.rootNode);

        return  bm && rn;
    }

    @Override
    public int hashCode() {
        return Objects.hash(buildMode, rootNode);
    }

    /**
     * This method reconnects given node to another node
     */
    public void remapNode(@NonNull String ip) {
        remapNode(getNodeByIp(ip));
    }

    /**
     * This method reconnects given node to another node
     */
    public void remapNode(@NonNull Node node) {
        synchronized (node) {

            node.getUpstreamNode().removeFromDownstreams(node);

            boolean m = false;
            for (val n: sortedNodes) {
                // we dont want to remap node to itself
                if (!Objects.equals(n, node) && n.status().equals(NodeStatus.ONLINE)) {
                    n.addDownstreamNode(node);
                    m = true;
                    break;
                }
            }

            // if we were unable to find good enough node - we'll map this node to the rootNode
            if (!m) {
                rootNode.addDownstreamNode(node);
            }

            // i hope we won't deadlock here? :)
            synchronized (this) {
                Collections.sort(sortedNodes);
            }
        }
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
    public boolean isKnownNode(@NonNull String ip) {
        return nodeMap.containsKey(ip);
    }

    /**
     * This method returns upstream connection for a given node
     */
    public Node getUpstreamForNode(@NonNull String ip) throws NoSuchElementException {
        val node = getNodeByIp(ip);

        return node.getUpstreamNode();
    }

    /**
     * This method returns downstream connections for a given node
     */
    public Collection<Node> getDownstreamsForNode(@NonNull String ip) throws NoSuchElementException {
        val node = getNodeByIp(ip);

        return node.getDownstreamNodes();
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
     * This method returns our mesh as collection of nodes
     * @return
     */
    protected Collection<Node> flatNodes() {
        return nodeMap.values();
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
    public Node getNodeByIp(@NonNull String ip) throws NoSuchElementException {
        val node = nodeMap.get(ip);
        if (node == null)
            throw new NoSuchElementException(ip);

        return node;
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

        @Getter(AccessLevel.PUBLIC)
        @Setter(AccessLevel.PROTECTED)
        @Builder.Default
        private boolean rootNode = false;

        @Getter
        private String id;

        @Getter
        private String ip;

        @Getter
        private int port;

        @Getter(AccessLevel.NONE)
        @Setter(AccessLevel.NONE)
        private Node upstream;

        @Getter(AccessLevel.NONE)
        @Setter(AccessLevel.NONE)
        private final List<Node> downstream = new ArrayList<>();

        private AtomicInteger position = new AtomicInteger(0);

        @Getter(AccessLevel.NONE)
        @Setter(AccessLevel.NONE)
        @Builder.Default
        private Atomic<NodeStatus> status = new Atomic<>(NodeStatus.ONLINE);

        /**
         * This method returns current status of this node
         * @return
         */
        public synchronized NodeStatus status() {
            return status.get();
        }

        /**
         * This method ret
         * @param status
         */
        protected synchronized void status(@NonNull NodeStatus status) {
            this.status.set(status);
        }

        /**
         * This method return candidate for new connection
         *
         * @param node
         * @return
         */
        protected Node getNextCandidate(Node node) {
            // if there's no candidates - just connect to this node
            if (downstream.size() == 0)
                return this;

            if (node == null)
                return downstream.get(0);

            // TODO: we can get rid of flat scan here, but it's one-off step anyway...

            // we return next node node after this node
            boolean b = false;
            for (val v: downstream) {
                if (b)
                    return v;

                if (Objects.equals(node, v))
                    b = true;
            }

            return null;
        }

        protected Node(boolean rootNode) {
            this.rootNode = rootNode;
        }

        /**
         * This method adds downstream node to the list of connections
         * @param node
         * @return
         */
        protected Node addDownstreamNode(@NonNull Node node) {
            this.downstream.add(node);
            node.setUpstreamNode(this);
            return node;
        }

        /**
         * This method pushes node to the bottom of this node downstream
         * @param node
         * @return
         */
        protected Node pushDownstreamNode(@NonNull Node node) {
            if (isRootNode()) {
                if (downstream.size() == 0) {
                    return addDownstreamNode(node);
                } else {
                    // we should find first not full sub-branch
                    for (val d: downstream)
                        if (d.numberOfDescendants() < MeshOrganizer.MAX_DEPTH * MeshOrganizer.MAX_DOWNSTREAMS)
                            return d.pushDownstreamNode(node);

                     // if we're here - we'll have to add new branch to the root
                    return addDownstreamNode(node);
                }
            } else {
                val distance = distanceFromRoot();

                for (val d: downstream)
                    if (d.numberOfDescendants() < MeshOrganizer.MAX_DOWNSTREAMS * (MeshOrganizer.MAX_DEPTH - distance))
                        return d.pushDownstreamNode(node);

                return addDownstreamNode(node);
            }
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
        public Node getUpstreamNode() {
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

        /**
         * This method returns number of hops between
         * @return
         */
        public int distanceFromRoot() {
            if (upstream.isRootNode())
                return 1;
            else
                return upstream.distanceFromRoot() + 1;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Node node = (Node) o;

            val rn = this.upstream == null ? "root" : this.upstream.getIp();
            val on = node.upstream == null ? "root" : node.upstream.getIp();

            return rootNode == node.rootNode &&
                    port == node.port &&
                    Objects.equals(id, node.id) &&
                    Objects.equals(ip, node.ip) &&
                    Objects.equals(downstream, node.downstream) &&
                    Objects.equals(status, node.status) &&
                    Objects.equals(rn, on);

        }

        @Override
        public int hashCode() {
            return Objects.hash(upstream == null ? "root" : upstream.getIp(), rootNode, id, ip, port, downstream, status);
        }

        /**
         * This method removes
         * @param node
         */
        protected synchronized void removeFromDownstreams(@NonNull Node node) {
            val r = downstream.remove(node);

            if (!r)
                throw new NoSuchElementException(node.getIp());
        }

        @Override
        public int compareTo(@NonNull Node o) {
            return Long.compare(this.numberOfDownstreams(), o.numberOfDownstreams());
        }
    }
}
