package org.nd4j.parameterserver.distributed.v2.transport.impl;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.ping.PingMessage;

import static org.junit.Assert.*;

@Slf4j
public class AeronUdpTransportTest {
    private static final String IP = "127.0.0.1";
    private static final int ROOT_PORT = 40781;

    @Test
    //@Ignore
    public void testBasic_Connection_1() throws Exception {
        // we definitely want to shutdown all transports after test, to avoid issues with shmem
        try(val transportA = new AeronUdpTransport(IP, ROOT_PORT, IP, ROOT_PORT, VoidConfiguration.builder().build());  val transportB = new AeronUdpTransport(IP, 40782, IP, ROOT_PORT, VoidConfiguration.builder().build())) {
            transportA.launchAsMaster();

            Thread.sleep(50);

            transportB.launch();

            Thread.sleep(50);

            assertEquals(2, transportA.getMesh().totalNodes());
            assertEquals(transportA.getMesh(), transportB.getMesh());
        }
    }
}