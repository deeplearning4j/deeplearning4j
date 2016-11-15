package org.nd4j.aeron.ipc.response;

import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import io.aeron.driver.ThreadingMode;
import org.agrona.CloseHelper;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.aeron.ipc.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.nd4j.linalg.factory.Nd4j.scalar;

/**
 * Created by agibsonccc on 10/3/16.
 */
public class AeronNDArrayResponseTest {
    private MediaDriver mediaDriver;
    private static Logger log = LoggerFactory.getLogger(NdArrayIpcTest.class);
    private Aeron.Context ctx;
    private Aeron.Context ctx2;

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
    public void testResponse() throws Exception {
        int streamId = 10;
        int responderStreamId = 11;
        String host = "127.0.0.1";
        Aeron aeron = Aeron.connect(getContext2());
        AeronNDArrayResponder responder = AeronNDArrayResponder.startSubscriber(
                aeron,
                host,
                40124,
                new NDArrayHolder() {
                    /**
                     * The number of updates
                     * that have been sent to this older.
                     *
                     * @return
                     */
                    @Override
                    public int totalUpdates() {
                        return 1;
                    }

                    /**
                     * Retrieve an ndarray
                     *
                     * @return
                     */
                    @Override
                    public INDArray get() {
                        return Nd4j.scalar(1.0);
                    }

                    /**
                     * Retrieve a partial view of the ndarray.
                     * This method uses tensor along dimension internally
                     * Note this will call dup()
                     *
                     * @param idx        the index of the tad to get
                     * @param dimensions the dimensions to use
                     * @return the tensor along dimension based on the index and dimensions
                     * from the master array.
                     */
                    @Override
                    public INDArray getTad(int idx, int... dimensions) {
                        return Nd4j.scalar(1.0);
                    }
                }

                ,responderStreamId);

        AtomicInteger count = new AtomicInteger(0);
        AtomicBoolean running = new AtomicBoolean(true);
        AeronNDArraySubscriber subscriber = AeronNDArraySubscriber.startSubscriber(
                aeron,
                host,
                40123,
                new NDArrayCallback() {
                    @Override
                    public void onNDArrayPartial(INDArray arr, long idx, int... dimensions) {
                        count.incrementAndGet();
                    }

                    @Override
                    public void onNDArray(INDArray arr) {
                        count.incrementAndGet();
                    }
                }
                , streamId, running);

        int expectedResponses = 10;
        HostPortPublisher publisher = HostPortPublisher
                .builder().aeron(aeron)
                .uriToSend(host + ":40123:" + streamId)
                .channel(AeronUtil
                        .aeronChannel(host,40124))
                .streamId(responderStreamId).build();

        Thread.sleep(10000);


        for(int i = 0; i < expectedResponses; i++) {
            publisher.send();
        }


        Thread.sleep(120000);


        assertEquals(expectedResponses,count.get());

        System.out.println("After");

        CloseHelper.close(responder);
        CloseHelper.close(subscriber);
        CloseHelper.close(publisher);
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
