package org.nd4j.parameterserver.client;

import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import org.agrona.CloseHelper;
import org.junit.*;
import org.nd4j.aeron.ipc.AeronUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.ParameterServerListener;
import org.nd4j.parameterserver.ParameterServerSubscriber;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static junit.framework.TestCase.assertFalse;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by agibsonccc on 10/3/16.
 */
public class ParameterServerClientTest {
    private static MediaDriver mediaDriver;
    private static Logger log = LoggerFactory.getLogger(ParameterServerClientTest.class);
    private static Aeron aeron;
    private static ParameterServerSubscriber masterNode,slaveNode;
    private static  int parameterLength = 1000;

    @BeforeClass
    public static void before() throws Exception {
        mediaDriver = MediaDriver.launchEmbedded(AeronUtil.getMediaDriverContext(parameterLength));
        System.setProperty("play.server.dir","/tmp");
        aeron = Aeron.connect(getContext());
        masterNode = new ParameterServerSubscriber(mediaDriver);
        masterNode.setAeron(aeron);
        masterNode.run(new String[] {
                "-m","true",
                "-s","1," + String.valueOf(parameterLength),
                "-p","40323",
                "-h","localhost",
                "-id","11",
                "-md", mediaDriver.aeronDirectoryName(),
                "-sp", "33000"
        });

        assertTrue(masterNode.isMaster());
        assertEquals(1000,masterNode.getParameterLength());
        assertEquals(40323,masterNode.getPort());
        assertEquals("localhost",masterNode.getHost());
        assertEquals(11,masterNode.getStreamId());
        assertEquals(12,masterNode.getResponder().getStreamId());

        slaveNode = new ParameterServerSubscriber(mediaDriver);
        slaveNode.setAeron(aeron);
        slaveNode.run(new String[] {
                "-l",String.valueOf(parameterLength),
                "-p","40426",
                "-h","localhost",
                "-id","10",
                "-pm",masterNode.getSubscriber().connectionUrl(),
                "-md", mediaDriver.aeronDirectoryName(),
                "-sp", "31000"
        });

        assertFalse(slaveNode.isMaster());
        assertEquals(1000,slaveNode.getParameterLength());
        assertEquals(40426,slaveNode.getPort());
        assertEquals("localhost",slaveNode.getHost());
        assertEquals(10,slaveNode.getStreamId());

        int tries = 10;
        while(!masterNode.subscriberLaunched() && !slaveNode.subscriberLaunched() && tries < 10) {
            Thread.sleep(10000);
            tries++;
        }

        if(!masterNode.subscriberLaunched() && !slaveNode.subscriberLaunched()) {
            throw new IllegalStateException("Failed to start master and slave node");
        }

        log.info("Using media driver directory " + mediaDriver.aeronDirectoryName());
        log.info("Launched media driver");
    }



    @Test
    public void testServer() throws Exception {
        ParameterServerClient client = ParameterServerClient
                .builder()
                .aeron(aeron)
                .ndarrayRetrieveUrl(masterNode.getResponder().connectionUrl())
                .ndarraySendUrl(slaveNode.getSubscriber().connectionUrl())
                .subscriberHost("localhost")
                .subscriberPort(40625)
                .subscriberStream(12).build();
        assertEquals("localhost:40625:12",client.connectionUrl());
        //flow 1:
        /**
         * Client (40125:12): sends array to listener on slave(40126:10)
         * which publishes to master (40123:11)
         * which adds the array for parameter averaging.
         * In this case totalN should be 1.
         */
        client.pushNDArray(Nd4j.ones(parameterLength));
        log.info("Pushed ndarray");
        Thread.sleep(30000);
        ParameterServerListener listener = (ParameterServerListener) masterNode.getCallback();
        assertEquals(1,listener.getTotalN().get());
        assertEquals(Nd4j.ones(parameterLength),listener.getArr());
        INDArray arr = client.getArray();
        assertEquals(Nd4j.ones(1000),arr);
    }

    @AfterClass
    public static void after() {
        if(mediaDriver != null)
            CloseHelper.quietClose(mediaDriver);
    }





    private static  Aeron.Context getContext() {
        return new Aeron.Context().publicationConnectionTimeout(-1)
                    .availableImageHandler(AeronUtil::printAvailableImage)
                    .unavailableImageHandler(AeronUtil::printUnavailableImage)
                    .aeronDirectoryName(mediaDriver.aeronDirectoryName()).keepAliveInterval(1000)
                    .errorHandler(e -> log.error(e.toString(), e));
    }


}
