package org.nd4j.parameterserver.distributed.v2.transport.impl;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.parameterserver.distributed.v2.messages.pairs.ping.PingMessage;

import static org.junit.Assert.*;

@Slf4j
public class AeronUdpTransportTest {

    @Test
    public void testBasic_Connection_1() throws Exception {
        // we definitely want to shutdown all transports after test, to avoid issues with shmem
        try(val transportA = new AeronUdpTransport();  val transportB = new AeronUdpTransport()) {
            transportA.launchAsMaster();
            transportB.launch();

            val ping = new PingMessage();

            transportA.sendMessageBlocking(ping, transportA.id());
        }
    }
}