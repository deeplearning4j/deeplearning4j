package org.nd4j.parameterserver.distributed;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class VoidParameterServerTest {
    private static List<String> localIPs;
    private static List<String> badIPs;

    @Before
    public void setUp() throws Exception {
        if (localIPs == null) {
            localIPs = new ArrayList<>(VoidParameterServer.getLocalAddresses());

            badIPs = Arrays.asList("127.0.0.1");
        }
    }

    @After
    public void tearDown() throws Exception {

    }

    @Test
    public void testNodeRole1() throws Exception {
        final Configuration conf = Configuration.builder()
                .port(34567)
                .numberOfShards(10)
                .shardAddresses(localIPs)
                .build();

        VoidParameterServer node = new VoidParameterServer();
        node.init(conf);

        assertEquals(NodeRole.SHARD, node.getNodeRole());
    }

    @Test
    public void testNodeRole2() throws Exception {
        final Configuration conf = Configuration.builder()
                .port(34567)
                .numberOfShards(10)
                .shardAddresses(badIPs)
                .backupAddresses(localIPs)
                .build();

        VoidParameterServer node = new VoidParameterServer();
        node.init(conf);

        assertEquals(NodeRole.BACKUP, node.getNodeRole());
    }

    @Test
    public void testNodeRole3() throws Exception {
        final Configuration conf = Configuration.builder()
                .port(34567)
                .numberOfShards(10)
                .shardAddresses(badIPs)
                .backupAddresses(badIPs)
                .build();

        VoidParameterServer node = new VoidParameterServer();
        node.init(conf);

        assertEquals(NodeRole.CLIENT, node.getNodeRole());
    }

    @Test
    public void testNodeInitialization1() throws Exception {
        final AtomicInteger failCnt = new AtomicInteger(0);

        final Configuration conf = Configuration.builder()
                .port(34567)
                .numberOfShards(10)
                .shardAddresses(localIPs)
                .build();

        Thread[] threads = new Thread[10];
        for (int t = 0; t < threads.length; t++) {
            threads[t] = new Thread(new Runnable() {
                @Override
                public void run() {
                    VoidParameterServer node = new VoidParameterServer();
                    node.init(conf);

                    if (node.getNodeRole() != NodeRole.SHARD)
                        failCnt.incrementAndGet();
                }
            });

            threads[t].start();
        }


        for (int t = 0; t < threads.length; t++) {
            threads[t].join();
        }

        assertEquals(0, failCnt.get());

    }
}