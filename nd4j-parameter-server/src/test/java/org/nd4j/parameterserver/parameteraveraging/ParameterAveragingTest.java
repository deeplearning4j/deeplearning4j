package org.nd4j.parameterserver.parameteraveraging;

import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import io.aeron.driver.ThreadingMode;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.aeron.ipc.AeronNDArrayPublisher;
import org.nd4j.aeron.ipc.AeronNDArraySubscriber;
import org.nd4j.aeron.ipc.AeronUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 9/28/16.
 */
public class ParameterAveragingTest {
    private static Logger log = LoggerFactory.getLogger(ParameterAveragingTest.class);
    private MediaDriver mediaDriver;
    private Aeron.Context ctx;

    @Before
    public void before() {
        final MediaDriver.Context ctx = new MediaDriver.Context()
                .threadingMode(ThreadingMode.DEDICATED)
                .dirsDeleteOnStart(true)
                .termBufferSparseFile(false)
                .conductorIdleStrategy(new BusySpinIdleStrategy())
                .receiverIdleStrategy(new BusySpinIdleStrategy())
                .senderIdleStrategy(new BusySpinIdleStrategy());
        mediaDriver = MediaDriver.launchEmbedded(ctx);
        System.out.println("Using media driver directory " + mediaDriver.aeronDirectoryName());
        System.out.println("Launched media driver");
    }


    @Test
    public void testAveragingPubSub() throws Exception {
        Aeron.Context ctx = new Aeron.Context().publicationConnectionTimeout(-1)
                .availableImageHandler(AeronUtil::printAvailableImage)
                .unavailableImageHandler(AeronUtil::printUnavailableImage)
                .aeronDirectoryName(mediaDriver.aeronDirectoryName()).keepAliveInterval(1000)
                .errorHandler(e -> log.error(e.toString(), e));
        AtomicBoolean running = new AtomicBoolean(true);
        int length = 1000;
        ParameterAveragingListener subscriber = new ParameterAveragingListener(length);
        AeronNDArraySubscriber subscriber1 =  AeronNDArraySubscriber.startSubscriber(ctx,"127.0.0.1",40123,subscriber,10,new AtomicBoolean(false));
        AeronNDArrayPublisher publisher =   AeronNDArrayPublisher.builder().streamId(10)
                .ctx(getContext()).channel("aeron:udp?endpoint=localhost:40123")
                .build();


        INDArray arr = Nd4j.ones(length);
        for(int i = 0; i< 1000 && running.get(); i++)
            publisher.publish(arr.dup());

        subscriber.finish();
        assertEquals(arr,subscriber.getArr());

        Thread.sleep(10000);
        publisher.close();
        running.set(false);



    }

    private Aeron.Context getContext() {
        if(ctx == null) ctx = new Aeron.Context().publicationConnectionTimeout(-1).availableImageHandler(AeronUtil::printAvailableImage)
                .unavailableImageHandler(AeronUtil::printUnavailableImage)
                .aeronDirectoryName(mediaDriver.aeronDirectoryName()).keepAliveInterval(1000)
                .errorHandler(e -> log.error(e.toString(), e));
        return ctx;
    }

}
