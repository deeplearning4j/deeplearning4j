package org.deeplearning4j.spark.models.sequencevectors.utils;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.net.util.SubnetUtils;
import org.deeplearning4j.spark.models.sequencevectors.primitives.NetworkInformation;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

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
    public NetworkOrganizer(@NonNull Collection<NetworkInformation> infoSet) {
        this(infoSet, null);
    }

    public NetworkOrganizer(@NonNull Collection<NetworkInformation> infoSet, String mask) {
        informationCollection = new ArrayList<>(infoSet);
        networkMask = mask;
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

        for (NetworkInformation information: informationCollection) {
            for (String ip: information.getIpAddresses()) {
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
        List<String> addresses = new ArrayList<>();
        /**
         * Since each ip address can be represented in 4-byte sequence, 1 byte per value, with leading order - we'll use that to build tree
         */

        for (NetworkInformation information: informationCollection) {
            for (String ip: information.getIpAddresses()) {
                // first we get binary representation for each IP
                String octet = convertIpToOctets(ip);

                // then we map each of them into virtual "tree", to find most popular networks within cluster
                tree.map(octet);
            }
        }

        return addresses;
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
            for(VirtualNode node: nodes.values()) {
                cnt.addAndGet(node.getNumDivergents());
            }
            return cnt.get();
        }

        public int getTotalBranches() {
            AtomicInteger cnt = new AtomicInteger(0);
            for(VirtualNode node: nodes.values()) {
                cnt.addAndGet(node.getCounter());
            }
            return cnt.get();
        }

        public String getHottestNetwork() {
            int max = 0;
            Character key = null;
            for (VirtualNode node: nodes.values()) {
                if (node.getCounter() > max) {
                    max = node.getCounter();
                    key = node.ownChar;
                }
            }
            log.info("top node: {} -> {}", key, max);
            VirtualNode topNode = nodes.get(key).getHottestNode(max);


           return topNode.rewind();
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
             for(VirtualNode node: nodes.values()) {
                 cnt.addAndGet(node.getNumDivergents());
             }
             return cnt.get();
        }


        protected int getDiscriminatedCount(){
            if (nodes.size() == 0 && counter == 1)
                return 0;

            AtomicInteger cnt = new AtomicInteger(Math.max(0, counter - 1));
            for(VirtualNode node: nodes.values()) {
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
            for (VirtualNode node: nodes.values()) {
                if (node.getCounter() >= threshold) {
                    log.info("    top node: {} -> {}", node.ownChar, node.getCounter());
                    return node.getHottestNode(threshold);
                }
            }

            log.info("No nodes below threshold");

            return this;
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
