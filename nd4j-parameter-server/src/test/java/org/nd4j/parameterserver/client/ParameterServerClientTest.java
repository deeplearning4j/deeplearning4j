package org.nd4j.parameterserver.client;

import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import io.aeron.driver.ThreadingMode;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.aeron.ipc.AeronNDArraySubscriber;
import org.nd4j.aeron.ipc.AeronUtil;
import org.nd4j.aeron.ipc.NDArrayHolder;
import org.nd4j.aeron.ipc.response.AeronNDArrayResponder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.parameteraveraging.ParameterAveragingListener;
import org.nd4j.parameterserver.parameteraveraging.ParameterAveragingSubscriber;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicInteger;

import static junit.framework.TestCase.assertFalse;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by agibsonccc on 10/3/16.
 */
public class ParameterServerClientTest {
    private MediaDriver mediaDriver;
    private static Logger log = LoggerFactory.getLogger(ParameterServerClientTest.class);
    private Aeron.Context ctx;
    private Aeron.Context ctx2;
    private ParameterAveragingSubscriber masterNode,slaveNode;
    private int parameterLength = 1000;
    @Before
    public void before() throws Exception {
        final MediaDriver.Context ctx = new MediaDriver.Context()
                .threadingMode(ThreadingMode.DEDICATED)
                .dirsDeleteOnStart(true)
                .termBufferSparseFile(false)
                .conductorIdleStrategy(new BusySpinIdleStrategy())
                .receiverIdleStrategy(new BusySpinIdleStrategy())
                .senderIdleStrategy(new BusySpinIdleStrategy());

        mediaDriver = MediaDriver.launchEmbedded(ctx);
        masterNode = new ParameterAveragingSubscriber(mediaDriver);
        masterNode.run(new String[] {
                "-m","true",
                "-l",String.valueOf(parameterLength),
                "-p","40123",
                "-h","localhost",
                "-id","11"
        });

        assertTrue(masterNode.isMaster());
        assertEquals(1000,masterNode.getParameterLength());
        assertEquals(40123,masterNode.getPort());
        assertEquals("localhost",masterNode.getHost());
        assertEquals(11,masterNode.getStreamId());
        assertEquals(12,masterNode.getResponder().getStreamId());

        slaveNode = new ParameterAveragingSubscriber(mediaDriver);
        slaveNode.run(new String[] {
                "-l",String.valueOf(parameterLength),
                "-p","40126",
                "-h","localhost",
                "-id","10",
                "-pm",masterNode.getSubscriber().connectionUrl()
        });

        assertFalse(slaveNode.isMaster());
        assertEquals(1000,slaveNode.getParameterLength());
        assertEquals(40126,slaveNode.getPort());
        assertEquals("localhost",slaveNode.getHost());
        assertEquals(10,slaveNode.getStreamId());


        while(!masterNode.subscriberLaunched() && !slaveNode.subscriberLaunched())
            Thread.sleep(10000);

        log.info("Using media driver directory " + mediaDriver.aeronDirectoryName());
        log.info("Launched media driver");
    }



    @Test
    public void testServer() throws Exception {
        ParameterServerClient client = ParameterServerClient
                .builder()
                .ctx(getContext())
                .ndarrayRetrieveUrl(masterNode.getResponder().connectionUrl())
                .ndarraySendUrl(slaveNode.getSubscriber().connectionUrl())
                .subscriberHost("localhost")
                .subscriberPort(40125)
                .subscriberStream(12).build();
        assertEquals("localhost:40125:12",client.connectionUrl());
        //flow 1:
        /**
         * Client (40125:12): sends array to listener on slave(40126:10)
         * which publishes to master (40123:11)
         * which adds the array for parameter averaging.
         * In this case totalN should be 1.
         */
        client.pushNDArray(Nd4j.ones(parameterLength));
        log.info("Pushed ndarray");
        Thread.sleep(10000);
        ParameterAveragingListener listener = (ParameterAveragingListener) masterNode.getCallback();
        assertEquals(1,listener.getTotalN().get());
        assertEquals(Nd4j.ones(parameterLength),listener.getArr());
        INDArray arr = client.getArray();
        assertEquals(Nd4j.ones(1000),arr);
    }

    private Aeron.Context getContext2() {
        if(ctx2 == null)
            ctx2 = new Aeron.Context().publicationConnectionTimeout(-1)
                    .availableImageHandler(AeronUtil::printAvailableImage)
                    .unavailableImageHandler(AeronUtil::printUnavailableImage)
                    .aeronDirectoryName(mediaDriver.aeronDirectoryName()).keepAliveInterval(1000)
                    .errorHandler(e -> log.error(e.toString(), e));
        return ctx2;
    }

    private Aeron.Context getContext() {
        if(ctx == null)
            ctx = new Aeron.Context().publicationConnectionTimeout(-1)
                    .availableImageHandler(AeronUtil::printAvailableImage)
                    .unavailableImageHandler(AeronUtil::printUnavailableImage)
                    .aeronDirectoryName(mediaDriver.aeronDirectoryName()).keepAliveInterval(1000)
                    .errorHandler(e -> log.error(e.toString(), e));
        return ctx;
    }


}
