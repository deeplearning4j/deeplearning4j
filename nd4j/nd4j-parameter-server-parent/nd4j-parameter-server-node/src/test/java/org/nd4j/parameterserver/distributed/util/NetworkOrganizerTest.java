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
import org.apache.commons.lang3.RandomUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.Timeout;

import java.util.*;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class NetworkOrganizerTest {
    @Before
    public void setUp() throws Exception {

    }

    @After
    public void tearDown() throws Exception {

    }

    @Rule
    public Timeout globalTimeout = Timeout.seconds(20); // 20 seconds max per method tested


    @Test
    public void testSimpleSelection1() throws Exception {
        NetworkOrganizer organizer = new NetworkOrganizer("127.0.0.0/24");
        List<String> list = organizer.getSubset(1);

        assertEquals(1, list.size());
        assertEquals("127.0.0.1", list.get(0));
    }

    @Test
    public void testSimpleSelection2() throws Exception {
        NetworkOrganizer organizer = new NetworkOrganizer("127.0.0.0/24");
        String ip = organizer.getMatchingAddress();

        assertEquals("127.0.0.1", ip);
    }

    @Test
    public void testSelectionUniformNetworkC1() {
        List<NetworkInformation> collection = new ArrayList<>();

        for (int i = 1; i < 128; i++) {
            NetworkInformation information = new NetworkInformation();

            information.addIpAddress("192.168.0." + i);
            information.addIpAddress(getRandomIp());

            collection.add(information);
        }

        NetworkOrganizer discoverer = new NetworkOrganizer(collection, "192.168.0.0/24");

        // check for primary subset (aka Shards)
        List<String> shards = discoverer.getSubset(10);

        assertEquals(10, shards.size());

        for (String ip : shards) {
            assertNotEquals(null, ip);
            assertTrue(ip.startsWith("192.168.0"));
        }


        // check for secondary subset (aka Backup)
        List<String> backup = discoverer.getSubset(10, shards);
        assertEquals(10, backup.size());
        for (String ip : backup) {
            assertNotEquals(null, ip);
            assertTrue(ip.startsWith("192.168.0"));
            assertFalse(shards.contains(ip));
        }
    }

    @Test
    public void testSelectionSingleBox1() throws Exception {
        List<NetworkInformation> collection = new ArrayList<>();
        NetworkInformation information = new NetworkInformation();
        information.addIpAddress("192.168.21.12");
        information.addIpAddress("10.0.27.19");
        collection.add(information);

        NetworkOrganizer organizer = new NetworkOrganizer(collection, "192.168.0.0/16");

        List<String> shards = organizer.getSubset(10);
        assertEquals(1, shards.size());
    }

    @Test
    public void testSelectionSingleBox2() throws Exception {
        List<NetworkInformation> collection = new ArrayList<>();
        NetworkInformation information = new NetworkInformation();
        information.addIpAddress("192.168.72.12");
        information.addIpAddress("10.2.88.19");
        collection.add(information);

        NetworkOrganizer organizer = new NetworkOrganizer(collection);

        List<String> shards = organizer.getSubset(10);
        assertEquals(1, shards.size());
    }


    @Test
    public void testSelectionDisjointNetworkC1() {
        List<NetworkInformation> collection = new ArrayList<>();

        for (int i = 1; i < 128; i++) {
            NetworkInformation information = new NetworkInformation();

            if (i < 20)
                information.addIpAddress("172.12.0." + i);

            information.addIpAddress(getRandomIp());

            collection.add(information);
        }

        NetworkOrganizer discoverer = new NetworkOrganizer(collection, "172.12.0.0/24");

        // check for primary subset (aka Shards)
        List<String> shards = discoverer.getSubset(10);

        assertEquals(10, shards.size());

        List<String> backup = discoverer.getSubset(10, shards);

        // we expect 9 here, thus backups will be either incomplete or complex sharding will be used for them

        assertEquals(9, backup.size());
        for (String ip : backup) {
            assertNotEquals(null, ip);
            assertTrue(ip.startsWith("172.12.0"));
            assertFalse(shards.contains(ip));
        }
    }


    /**
     * In this test we'll check shards selection in "casual" AWS setup
     * By default AWS box has only one IP from 172.16.0.0/12 space + local loopback IP, which isn't exposed
     *
     * @throws Exception
     */
    @Test
    public void testSelectionWithoutMaskB1() throws Exception {
        List<NetworkInformation> collection = new ArrayList<>();

        // we imitiate 512 cluster nodes here
        for (int i = 0; i < 512; i++) {
            NetworkInformation information = new NetworkInformation();

            information.addIpAddress(getRandomAwsIp());
            collection.add(information);
        }

        NetworkOrganizer organizer = new NetworkOrganizer(collection);

        List<String> shards = organizer.getSubset(10);

        assertEquals(10, shards.size());

        List<String> backup = organizer.getSubset(10, shards);

        assertEquals(10, backup.size());
        for (String ip : backup) {
            assertNotEquals(null, ip);
            assertTrue(ip.startsWith("172."));
            assertFalse(shards.contains(ip));
        }
    }

    /**
     * In this test we check for environment which has AWS-like setup:
     *  1) Each box has IP address from 172.16.0.0/12 range
     *  2) Within original homogenous network, we have 3 separate networks:
     *      A) 192.168.0.X
     *      B) 10.0.12.X
     *      C) 10.172.12.X
     *
     * @throws Exception
     */
    @Test
    public void testSelectionWithoutMaskB2() throws Exception {
        List<NetworkInformation> collection = new ArrayList<>();

        // we imitiate 512 cluster nodes here
        for (int i = 0; i < 512; i++) {
            NetworkInformation information = new NetworkInformation();

            information.addIpAddress(getRandomAwsIp());

            if (i < 30) {
                information.addIpAddress("192.168.0." + i);
            } else if (i < 95) {
                information.addIpAddress("10.0.12." + i);
            } else if (i < 255) {
                information.addIpAddress("10.172.12." + i);
            }

            collection.add(information);
        }

        NetworkOrganizer organizer = new NetworkOrganizer(collection);

        List<String> shards = organizer.getSubset(15);

        assertEquals(15, shards.size());

        for (String ip : shards) {
            assertNotEquals(null, ip);
            assertTrue(ip.startsWith("172."));
        }

        List<String> backup = organizer.getSubset(15, shards);

        for (String ip : backup) {
            assertNotEquals(null, ip);
            assertTrue(ip.startsWith("172."));
            assertFalse(shards.contains(ip));
        }

    }


    /**
     * Here we just check formatting for octets
     */
    @Test
    public void testFormat1() throws Exception {
        for (int i = 0; i < 256; i++) {
            String octet = NetworkOrganizer.toBinaryOctet(i);
            assertEquals(8, octet.length());
            log.trace("i: {}; Octet: {}", i, octet);
        }
    }


    @Test
    public void testFormat2() throws Exception {
        for (int i = 0; i < 1000; i++) {
            String octets = NetworkOrganizer.convertIpToOctets(getRandomIp());

            // we just expect 8 bits per bloc, 4 blocks in total, plus 3 dots between blocks
            assertEquals(35, octets.length());
        }
    }


    @Test
    public void testNetTree1() throws Exception {
        List<String> ips = Arrays.asList("192.168.0.1", "192.168.0.2");

        NetworkOrganizer.VirtualTree tree = new NetworkOrganizer.VirtualTree();

        for (String ip : ips)
            tree.map(NetworkOrganizer.convertIpToOctets(ip));

        assertEquals(2, tree.getUniqueBranches());
        assertEquals(2, tree.getTotalBranches());

        log.info("rewind: {}", tree.getHottestNetwork());
    }

    @Test
    public void testNetTree2() throws Exception {
        List<String> ips = Arrays.asList("192.168.12.2", "192.168.0.2", "192.168.0.2", "192.168.62.92");

        NetworkOrganizer.VirtualTree tree = new NetworkOrganizer.VirtualTree();

        for (String ip : ips)
            tree.map(NetworkOrganizer.convertIpToOctets(ip));

        assertEquals(3, tree.getUniqueBranches());
        assertEquals(4, tree.getTotalBranches());
    }

    /**
     * This test is just a naive test for counters
     *
     * @throws Exception
     */
    @Test
    public void testNetTree3() throws Exception {
        List<String> ips = new ArrayList<>();

        NetworkOrganizer.VirtualTree tree = new NetworkOrganizer.VirtualTree();

        for (int i = 0; i < 3000; i++)
            ips.add(getRandomIp());


        for (int i = 0; i < 20; i++)
            ips.add("192.168.12." + i);

        Collections.shuffle(ips);

        Set<String> uniqueIps = new HashSet<>(ips);

        for (String ip : uniqueIps)
            tree.map(NetworkOrganizer.convertIpToOctets(ip));

        assertEquals(uniqueIps.size(), tree.getTotalBranches());
        assertEquals(uniqueIps.size(), tree.getUniqueBranches());

    }

    @Test
    public void testNetTree4() throws Exception {
        List<String> ips = Arrays.asList("192.168.12.2", "192.168.0.2", "192.168.0.2", "192.168.62.92", "5.3.4.5");

        NetworkOrganizer.VirtualTree tree = new NetworkOrganizer.VirtualTree();

        for (String ip : ips)
            tree.map(NetworkOrganizer.convertIpToOctets(ip));

        assertEquals(4, tree.getUniqueBranches());
        assertEquals(5, tree.getTotalBranches());
    }

    @Test
    public void testNetTree5() throws Exception {
        List<String> ips = new ArrayList<>();

        NetworkOrganizer.VirtualTree tree = new NetworkOrganizer.VirtualTree();

        for (int i = 0; i < 254; i++)
            ips.add(getRandomIp());


        for (int i = 1; i < 255; i++)
            ips.add("192.168.12." + i);

        Collections.shuffle(ips);

        Set<String> uniqueIps = new HashSet<>(ips);

        for (String ip : uniqueIps)
            tree.map(NetworkOrganizer.convertIpToOctets(ip));

        assertEquals(508, uniqueIps.size());

        assertEquals(uniqueIps.size(), tree.getTotalBranches());
        assertEquals(uniqueIps.size(), tree.getUniqueBranches());

        /**
         * Now the most important part here. we should get 192.168.12. as the most "popular" branch
         */

        String networkA = tree.getHottestNetworkA();

        assertEquals("11000000", networkA);

        String networkAB = tree.getHottestNetworkAB();

        //        assertEquals("11000000.10101000", networkAB);
    }

    @Test
    public void testNetTree6() throws Exception {
        List<String> ips = new ArrayList<>();

        NetworkOrganizer.VirtualTree tree = new NetworkOrganizer.VirtualTree();

        for (int i = 0; i < 254; i++)
            ips.add(getRandomIp());


        for (int i = 1; i < 255; i++)
            ips.add(getRandomAwsIp());

        Collections.shuffle(ips);

        Set<String> uniqueIps = new HashSet<>(ips);

        for (String ip : uniqueIps)
            tree.map(NetworkOrganizer.convertIpToOctets(ip));

        assertEquals(508, uniqueIps.size());

        assertEquals(uniqueIps.size(), tree.getTotalBranches());
        assertEquals(uniqueIps.size(), tree.getUniqueBranches());

        /**
         * Now the most important part here. we should get 192.168.12. as the most "popular" branch
         */

        String networkA = tree.getHottestNetworkA();

        assertEquals("10101100", networkA);

        String networkAB = tree.getHottestNetworkAB();

        //  assertEquals("10101100.00010000", networkAB);
    }

    protected String getRandomIp() {
        StringBuilder builder = new StringBuilder();

        builder.append(RandomUtils.nextInt(1, 172)).append(".");
        builder.append(RandomUtils.nextInt(0, 255)).append(".");
        builder.append(RandomUtils.nextInt(0, 255)).append(".");
        builder.append(RandomUtils.nextInt(1, 255));

        return builder.toString();
    }

    protected String getRandomAwsIp() {
        StringBuilder builder = new StringBuilder("172.");

        builder.append(RandomUtils.nextInt(16, 32)).append(".");
        builder.append(RandomUtils.nextInt(0, 255)).append(".");
        builder.append(RandomUtils.nextInt(1, 255));

        return builder.toString();
    }
}
