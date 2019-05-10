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

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.net.util.SubnetUtils;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.net.InterfaceAddress;
import java.net.NetworkInterface;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Utility class that provides
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class NetworkOrganizer {
    protected List<NetworkInformation> informationCollection;
    protected String networkMask;
    protected VirtualTree tree = new VirtualTree();

    /**
     * This constructor is NOT implemented yet
     *
     * @param infoSet
     */
    // TODO: implement this one properly, we should build mask out of list of Ips
    protected NetworkOrganizer(@NonNull Collection<NetworkInformation> infoSet) {
        this(infoSet, null);
    }

    public NetworkOrganizer(@NonNull Collection<NetworkInformation> infoSet, String mask) {
        informationCollection = new ArrayList<>(infoSet);
        networkMask = mask;
    }


    /**
     * This constructor builds format from own
     *
     * @param networkMask
     */
    public NetworkOrganizer(@NonNull String networkMask) {
        this.informationCollection = buildLocalInformation();
        this.networkMask = networkMask;
    }

    protected List<NetworkInformation> buildLocalInformation() {
        List<NetworkInformation> list = new ArrayList<>();
        NetworkInformation netInfo = new NetworkInformation();
        try {
            List<NetworkInterface> interfaces = Collections.list(NetworkInterface.getNetworkInterfaces());

            for (NetworkInterface networkInterface : interfaces) {
                if (!networkInterface.isUp())
                    continue;

                for (InterfaceAddress address : networkInterface.getInterfaceAddresses()) {
                    String addr = address.getAddress().getHostAddress();

                    if (addr == null || addr.isEmpty() || addr.contains(":"))
                        continue;

                    netInfo.getIpAddresses().add(addr);
                }
            }
            list.add(netInfo);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return list;
    }

    /**
     * This method returns local IP address that matches given network mask.
     * To be used with single-argument constructor only.
     *
     * @return
     */
    public String getMatchingAddress() {
        if (informationCollection.size() > 1)
            this.informationCollection = buildLocalInformation();

        List<String> list = getSubset(1);
        if (list.size() < 1)
            throw new ND4JIllegalStateException(
                            "Unable to find network interface matching requested mask: " + networkMask);

        if (list.size() > 1)
            log.warn("We have {} local IPs matching given netmask [{}]", list.size(), networkMask);

        return list.get(0);
    }

    /**
     * This method returns specified number of IP addresses from original list of addresses
     *
     * @param numShards
     * @return
     */
    public List<String> getSubset(int numShards) {
        return getSubset(numShards, null);
    }


    /**
     * This method returns specified number of IP addresses from original list of addresses, that are NOT listen in primary collection
     *
     * @param numShards
     * @param primary Collection of IP addresses that shouldn't be in result
     * @return
     */
    public List<String> getSubset(int numShards, Collection<String> primary) {
        /**
         * If netmask in unset, we'll use manual
         */
        if (networkMask == null)
            return getIntersections(numShards, primary);

        List<String> addresses = new ArrayList<>();

        SubnetUtils utils = new SubnetUtils(networkMask);

        Collections.shuffle(informationCollection);

        for (NetworkInformation information : informationCollection) {
            for (String ip : information.getIpAddresses()) {
                if (primary != null && primary.contains(ip))
                    continue;

                if (utils.getInfo().isInRange(ip)) {
                    log.debug("Picked {} as {}", ip, primary == null ? "Shard" : "Backup");
                    addresses.add(ip);
                }

                if (addresses.size() >= numShards)
                    break;
            }

            if (addresses.size() >= numShards)
                break;
        }

        return addresses;
    }

    protected static String convertIpToOctets(@NonNull String ip) {
        String[] octets = ip.split("\\.");
        if (octets.length != 4)
            throw new UnsupportedOperationException();

        StringBuilder builder = new StringBuilder();

        for (int i = 0; i < 3; i++)
            builder.append(toBinaryOctet(octets[i])).append(".");
        builder.append(toBinaryOctet(octets[3]));

        return builder.toString();
    }

    protected static String toBinaryOctet(@NonNull Integer value) {
        if (value < 0 || value > 255)
            throw new ND4JIllegalStateException("IP octets cant hold values below 0 or above 255");
        String octetBase = Integer.toBinaryString(value);
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < 8 - octetBase.length(); i++) {
            builder.append("0");
        }
        builder.append(octetBase);

        return builder.toString();
    }

    protected static String toBinaryOctet(@NonNull String value) {
        return toBinaryOctet(Integer.parseInt(value));
    }

    /**
     * This method returns specified numbers of IP's by parsing original list of trees into some form of binary tree
     *
     * @param numShards
     * @param primary
     * @return
     */
    protected List<String> getIntersections(int numShards, Collection<String> primary) {
        /**
         * Since each ip address can be represented in 4-byte sequence, 1 byte per value, with leading order - we'll use that to build tree
         */
        if (primary == null) {
            for (NetworkInformation information : informationCollection) {
                for (String ip : information.getIpAddresses()) {
                    // first we get binary representation for each IP
                    String octet = convertIpToOctets(ip);

                    // then we map each of them into virtual "tree", to find most popular networks within cluster
                    tree.map(octet);
                }
            }

            // we get most "popular" A network from tree now
            String octetA = tree.getHottestNetworkA();

            List<String> candidates = new ArrayList<>();

            AtomicInteger matchCount = new AtomicInteger(0);
            for (NetworkInformation node : informationCollection) {
                for (String ip : node.getIpAddresses()) {
                    String octet = convertIpToOctets(ip);

                    // calculating matches
                    if (octet.startsWith(octetA)) {
                        matchCount.incrementAndGet();
                        candidates.add(ip);
                        break;
                    }
                }
            }

            /**
             * TODO: improve this. we just need to iterate over popular networks instead of single top A network
             */
            if (matchCount.get() != informationCollection.size())
                throw new ND4JIllegalStateException("Mismatching A class");

            Collections.shuffle(candidates);

            return new ArrayList<>(candidates.subList(0, Math.min(numShards, candidates.size())));
        } else {
            // if primary isn't null, we expect network to be already filtered
            String octetA = tree.getHottestNetworkA();

            List<String> candidates = new ArrayList<>();

            for (NetworkInformation node : informationCollection) {
                for (String ip : node.getIpAddresses()) {
                    String octet = convertIpToOctets(ip);

                    // calculating matches
                    if (octet.startsWith(octetA) && !primary.contains(ip)) {
                        candidates.add(ip);
                        break;
                    }
                }
            }

            Collections.shuffle(candidates);

            return new ArrayList<>(candidates.subList(0, Math.min(numShards, candidates.size())));
        }
    }



    public static class VirtualTree {
        // funny but we'll have max of 2 sub-nodes on node
        protected Map<Character, VirtualNode> nodes = new HashMap<>();

        /**
         * PLEASE NOTE: This method expects binary octets inside the string argument
         *
         * @param string
         */
        public void map(@NonNull String string) {
            String[] chars = string.split("");
            Character ch = chars[0].charAt(0);

            if (ch.charValue() != '0' && ch.charValue() != '1')
                throw new ND4JIllegalStateException("VirtualTree expects binary octets as input");

            if (!nodes.containsKey(ch))
                nodes.put(ch, new VirtualNode(ch, null));

            nodes.get(ch).map(chars, 1);
        }

        public int getUniqueBranches() {
            AtomicInteger cnt = new AtomicInteger(nodes.size());
            for (VirtualNode node : nodes.values()) {
                cnt.addAndGet(node.getNumDivergents());
            }
            return cnt.get();
        }

        public int getTotalBranches() {
            AtomicInteger cnt = new AtomicInteger(0);
            for (VirtualNode node : nodes.values()) {
                cnt.addAndGet(node.getCounter());
            }
            return cnt.get();
        }

        public String getHottestNetwork() {
            int max = 0;
            Character key = null;
            for (VirtualNode node : nodes.values()) {
                if (node.getCounter() > max) {
                    max = node.getCounter();
                    key = node.ownChar;
                }
            }
            VirtualNode topNode = nodes.get(key).getHottestNode(max);


            return topNode.rewind();
        }

        protected VirtualNode getHottestNode() {
            int max = 0;
            Character key = null;
            for (VirtualNode node : nodes.values()) {
                if (node.getCounter() > max) {
                    max = node.getCounter();
                    key = node.ownChar;
                }
            }

            return nodes.get(key);
        }

        public String getHottestNetworkA() {
            StringBuilder builder = new StringBuilder();

            int depth = 0;
            VirtualNode startingNode = getHottestNode();

            if (startingNode == null)
                throw new ND4JIllegalStateException(
                                "VirtualTree wasn't properly initialized, and doesn't have any information within");

            builder.append(startingNode.ownChar);

            for (int i = 0; i < 7; i++) {
                startingNode = startingNode.getHottestNode();
                builder.append(startingNode.ownChar);
            }

            return builder.toString();
        }

        /**
         * This method returns FULL A octet + B octet UP TO FIRST SIGNIFICANT BIT
         * @return
         */
        public String getHottestNetworkAB() {
            StringBuilder builder = new StringBuilder();

            int depth = 0;
            VirtualNode startingNode = getHottestNode();

            if (startingNode == null)
                throw new ND4JIllegalStateException(
                                "VirtualTree wasn't properly initialized, and doesn't have any information within");

            builder.append(startingNode.ownChar);

            // building first octet
            for (int i = 0; i < 7; i++) {
                startingNode = startingNode.getHottestNode();
                builder.append(startingNode.ownChar);
            }

            // adding dot after first octet
            startingNode = startingNode.getHottestNode();
            builder.append(startingNode.ownChar);

            // building partial octet for subnet B
            /**
             * basically we want widest possible match here
             */
            for (int i = 0; i < 8; i++) {

            }

            return builder.toString();
        }
    }


    public static class VirtualNode {
        protected Map<Character, VirtualNode> nodes = new HashMap<>();
        protected final Character ownChar;
        protected int counter = 0;
        protected VirtualNode parentNode;

        public VirtualNode(Character character, VirtualNode parentNode) {
            this.ownChar = character;
            this.parentNode = parentNode;
        }

        public void map(String[] chars, int position) {
            counter++;
            if (position < chars.length) {
                Character ch = chars[position].charAt(0);
                if (!nodes.containsKey(ch))
                    nodes.put(ch, new VirtualNode(ch, this));

                nodes.get(ch).map(chars, position + 1);
            }
        }

        protected int getNumDivergents() {
            if (nodes.size() == 0)
                return 0;

            AtomicInteger cnt = new AtomicInteger(nodes.size() - 1);
            for (VirtualNode node : nodes.values()) {
                cnt.addAndGet(node.getNumDivergents());
            }
            return cnt.get();
        }


        protected int getDiscriminatedCount() {
            if (nodes.size() == 0 && counter == 1)
                return 0;

            AtomicInteger cnt = new AtomicInteger(Math.max(0, counter - 1));
            for (VirtualNode node : nodes.values()) {
                cnt.addAndGet(node.getDiscriminatedCount());
            }
            return cnt.get();
        }

        protected int getCounter() {
            return counter;
        }

        /**
         * This method returns most popular sub-node
         * @return
         */
        protected VirtualNode getHottestNode(int threshold) {
            for (VirtualNode node : nodes.values()) {
                if (node.getCounter() >= threshold) {
                    return node.getHottestNode(threshold);
                }
            }

            return this;
        }

        protected VirtualNode getHottestNode() {
            int max = 0;
            Character ch = null;
            for (VirtualNode node : nodes.values()) {
                if (node.getCounter() > max) {
                    ch = node.ownChar;
                    max = node.getCounter();
                }
            }

            return nodes.get(ch);
        }

        protected String rewind() {
            StringBuilder builder = new StringBuilder();

            VirtualNode lastNode = this;
            while ((lastNode = lastNode.parentNode) != null) {
                builder.append(lastNode.ownChar);
            }

            return builder.reverse().toString();
        }
    }
}
