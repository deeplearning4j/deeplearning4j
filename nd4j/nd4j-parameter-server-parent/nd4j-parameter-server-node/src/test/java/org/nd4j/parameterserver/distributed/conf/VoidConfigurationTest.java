package org.nd4j.parameterserver.distributed.conf;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.Timeout;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class VoidConfigurationTest {

    @Rule
    public Timeout globalTimeout = Timeout.seconds(30);

    @Test
    public void testNetworkMask1() throws Exception {
        VoidConfiguration configuration = new VoidConfiguration();
        configuration.setNetworkMask("192.168.1.0/24");

        assertEquals("192.168.1.0/24", configuration.getNetworkMask());
    }


    @Test
    public void testNetworkMask2() throws Exception {
        VoidConfiguration configuration = new VoidConfiguration();
        configuration.setNetworkMask("192.168.1.12");

        assertEquals("192.168.1.0/24", configuration.getNetworkMask());
    }

    @Test
    public void testNetworkMask5() throws Exception {
        VoidConfiguration configuration = new VoidConfiguration();
        configuration.setNetworkMask("192.168.0.0/16");

        assertEquals("192.168.0.0/16", configuration.getNetworkMask());
    }

    @Test
    public void testNetworkMask6() throws Exception {
        VoidConfiguration configuration = new VoidConfiguration();
        configuration.setNetworkMask("192.168.0.0/8");

        assertEquals("192.168.0.0/8", configuration.getNetworkMask());
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testNetworkMask3() throws Exception {
        VoidConfiguration configuration = new VoidConfiguration();
        configuration.setNetworkMask("192.256.1.1/24");

        assertEquals("192.168.1.0/24", configuration.getNetworkMask());
    }

    @Test(expected = ND4JIllegalStateException.class)
    public void testNetworkMask4() throws Exception {
        VoidConfiguration configuration = new VoidConfiguration();
        configuration.setNetworkMask("0.0.0.0/8");

        assertEquals("192.168.1.0/24", configuration.getNetworkMask());
    }
}
