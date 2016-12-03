package org.nd4j.parameterserver.node;

import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import org.agrona.CloseHelper;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.aeron.ipc.AeronUtil;
import org.nd4j.aeron.ipc.NDArrayMessage;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.ParameterServerSubscriber;
import org.nd4j.parameterserver.client.ParameterServerClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Created by agibsonccc on 12/3/16.
 */
public class ParameterServerNodeTest {
    private static MediaDriver mediaDriver;
    private static Logger log = LoggerFactory.getLogger(ParameterServerNodeTest.class);
    private static Aeron aeron;
    private static ParameterServerNode parameterServerNode;
    private static int parameterLength = 4;

    @BeforeClass
    public static void before() throws Exception {
        mediaDriver = MediaDriver.launchEmbedded(AeronUtil.getMediaDriverContext(parameterLength));
        System.setProperty("play.server.dir","/tmp");
        aeron = Aeron.connect(getContext());
        parameterServerNode = new ParameterServerNode(mediaDriver);
        parameterServerNode.runMain(new String[] {
                "-m","true",
                "-s","1," + String.valueOf(parameterLength),
                "-p","40323",
                "-h","localhost",
                "-id","11",
                "-md", mediaDriver.aeronDirectoryName(),
                "-sp", "9000",
                "-sh","localhost"
        });

        while(!parameterServerNode.getSubscriber().subscriberLaunched()) {
            Thread.sleep(10000);
        }

    }

    @Test
    public void testSimulateRun() throws Exception {
        int numCores = Runtime.getRuntime().availableProcessors();
        ExecutorService executorService = Executors.newFixedThreadPool(numCores);
        ParameterServerClient[] clients = new ParameterServerClient[numCores];
        String host = "localhost";
        for(int i = 0; i < numCores; i++) {
            clients[i] = ParameterServerClient.builder()
                    .aeron(aeron).masterStatusHost(host)
                    .masterStatusPort(9000).subscriberHost(host).subscriberPort(40325 + i).subscriberStream(10 + i)
                    .ndarrayRetrieveUrl(parameterServerNode.getSubscriber().getResponder().connectionUrl())
                    .ndarraySendUrl(parameterServerNode.getSubscriber().getSubscriber().connectionUrl())
                    .build();
        }

        Thread.sleep(60000);

        //no arrays have been sent yet
        for(int i = 0; i < numCores; i++) {
            assertFalse(clients[i].isReadyForNext());
        }

        //send "numCores" arrays, the default parameter server updater
        //is synchronous so it should be "ready" when number of updates == number of workers
        for(int i = 0; i < numCores; i++) {
            clients[i].pushNDArrayMessage(NDArrayMessage.wholeArrayUpdate(Nd4j.ones(parameterLength)));
        }

        Thread.sleep(10000);

        //all arrays should have been sent
        for(int i = 0; i < numCores; i++) {
            assertTrue(clients[i].isReadyForNext());
        }

        Thread.sleep(10000);

        for(int i = 0; i < 1; i++) {
            assertEquals(Nd4j.valueArrayOf(1,parameterLength,numCores),clients[i].getArray());
            Thread.sleep(1000);
        }

        executorService.shutdown();

        Thread.sleep(60000);

        parameterServerNode.stop();


    }


    private static  Aeron.Context getContext() {
        return new Aeron.Context().publicationConnectionTimeout(-1)
                .availableImageHandler(AeronUtil::printAvailableImage)
                .unavailableImageHandler(AeronUtil::printUnavailableImage)
                .aeronDirectoryName(mediaDriver.aeronDirectoryName()).keepAliveInterval(1000)
                .errorHandler(e -> log.error(e.toString(), e));
    }


}
