package org.deeplearning4j.spark.models.sequencevectors.utils;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.net.util.SubnetUtils;
import org.deeplearning4j.spark.models.sequencevectors.primitives.NetworkInformation;

import java.util.*;

/**
 * Utility class that provides
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class NetworkDiscoverer {
    protected List<NetworkInformation> informationCollection;
    protected String networkMask;

    /**
     * This constructor is NOT implemented yet
     *
     * @param infoSet
     */
    // TODO: implement this one properly, we should build mask out of list of Ips
    public NetworkDiscoverer(@NonNull Collection<NetworkInformation> infoSet) {
        this(infoSet, null);
    }

    public NetworkDiscoverer(@NonNull Collection<NetworkInformation> infoSet, String mask) {
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
}
