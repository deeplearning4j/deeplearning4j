package org.deeplearning4j.spark.models.sequencevectors.utils;

import org.apache.commons.lang3.RandomUtils;
import org.deeplearning4j.spark.models.sequencevectors.primitives.NetworkInformation;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class NetworkDiscovererTest {
    @Before
    public void setUp() throws Exception {

    }

    @After
    public void tearDown() throws Exception {

    }

    @Test
    public void testSelectionUniformNetworkC1() {
        List<NetworkInformation> collection = new ArrayList<>();

        for(int i = 1; i < 128; i++) {
            NetworkInformation information = new NetworkInformation();

            information.addIpAddress("192.168.0." + i);
            information.addIpAddress(getRandomIp());

            collection.add(information);
        }

        NetworkDiscoverer discoverer = new NetworkDiscoverer(collection, "192.168.0.0/24");

        // check for primary subset (aka Shards)
        List<String> shards = discoverer.getSubset(10);

        assertEquals(10, shards.size());

        for (String ip: shards) {
            assertNotEquals(null, ip);
            assertTrue(ip.startsWith("192.168.0"));
        }


        // check for secondary subset (aka Backup)
        List<String> backup = discoverer.getSubset(10, shards);
        assertEquals(10, backup.size());
        for (String ip: backup) {
            assertNotEquals(null, ip);
            assertTrue(ip.startsWith("192.168.0"));
            assertFalse(shards.contains(ip));
        }
    }


    @Test
    public void testSelectionDisjointNetworkC1() {
        List<NetworkInformation> collection = new ArrayList<>();

        for(int i = 1; i < 128; i++) {
            NetworkInformation information = new NetworkInformation();

            if (i < 20)
                information.addIpAddress("172.12.0." + i);

            information.addIpAddress(getRandomIp());

            collection.add(information);
        }

        NetworkDiscoverer discoverer = new NetworkDiscoverer(collection, "172.12.0.0/24");

        // check for primary subset (aka Shards)
        List<String> shards = discoverer.getSubset(10);

        assertEquals(10, shards.size());

        List<String> backup = discoverer.getSubset(10, shards);

        // we expect 9 here, thus backups will be either incomplete or complex sharding will be used for them

        assertEquals(9, backup.size());
        for (String ip: backup) {
            assertNotEquals(null, ip);
            assertTrue(ip.startsWith("172.12.0"));
            assertFalse(shards.contains(ip));
        }
    }


    protected String getRandomIp() {
        StringBuilder builder = new StringBuilder();

        builder.append(RandomUtils.nextInt(1, 172)).append(".");
        builder.append(RandomUtils.nextInt(1, 255)).append(".");
        builder.append(RandomUtils.nextInt(1, 255)).append(".");
        builder.append(RandomUtils.nextInt(1, 255));

        return builder.toString();
    }
}