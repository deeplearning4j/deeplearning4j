package org.nd4j.parameterserver.parameteraveraging;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import com.google.common.base.Preconditions;
import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import io.aeron.driver.ThreadingMode;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.nd4j.aeron.ipc.AeronConnectionInformation;
import org.nd4j.aeron.ipc.AeronNDArraySubscriber;
import org.nd4j.aeron.ipc.AeronUtil;
import org.nd4j.aeron.ipc.NDArrayCallback;
import org.nd4j.aeron.ipc.response.AeronNDArrayResponder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Subscriber main class for
 * the parameter
 * averaging server
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
@Data
public class ParameterAveragingSubscriber {

    private static Logger log = LoggerFactory.getLogger(ParameterAveragingSubscriber.class);

    @Parameter(names={"-p","--port"}, description = "The port to listen on for the daemon", arity = 1)
    private int port = 40123;
    @Parameter(names={"-id","--streamId"}, description = "The stream id to listen on", arity = 1)
    private int streamId = 10;
    @Parameter(names={"-h","--host"}, description = "Host for the server to bind to", arity = 1)
    private String host = "localhost";
    @Parameter(names={"-l","--parameterLength"}, description = "Parameter length for parameter averaging", arity = 1)
    private int parameterLength = 1000;
    @Parameter(names={"-d","--deleteDirectoryOnStart"}, description = "Delete aeron directory on startup.", arity = 1)
    private boolean deleteDirectoryOnStart = true;
    @Parameter(names={"-m","--master"}, description = "Whether this subscriber is a master node or not.", arity = 1)
    private boolean master = false;
    @Parameter(names={"-pm","--publishmaster"}, description = "Publish master url: host:port - this is for peer nodes needing to publish to another peer.", arity = 1)
    private String publishMasterUrl = "localhost:40123";

    private MediaDriver mediaDriver;
    private AeronNDArrayResponder responder;
    private AeronNDArraySubscriber subscriber;
    private   NDArrayCallback callback;
    /**
     * Allow passing in a
     * media driver that already exists
     * @param mediaDriver
     */
    public ParameterAveragingSubscriber(MediaDriver mediaDriver) {
        Preconditions.checkNotNull(mediaDriver);
        this.mediaDriver = mediaDriver;
    }

    /**
     *
     * @param args
     */
    public void run(String[] args) {
        JCommander jcmdr = new JCommander(this);

        try {
            jcmdr.parse(args);
        } catch(ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try{ Thread.sleep(500); } catch(Exception e2){ }
            throw e;
        }

        if(publishMasterUrl == null && !master)
            throw new IllegalStateException("Please specify a master url or set master to true");

        //allows passing in a media driver for things like uni tests
        if(mediaDriver == null) {
            final MediaDriver.Context mediaDriverCtx = new MediaDriver.Context()
                    .threadingMode(ThreadingMode.DEDICATED)
                    .dirsDeleteOnStart(deleteDirectoryOnStart)
                    .termBufferSparseFile(false)
                    .conductorIdleStrategy(new BusySpinIdleStrategy())
                    .receiverIdleStrategy(new BusySpinIdleStrategy())
                    .senderIdleStrategy(new BusySpinIdleStrategy());

            mediaDriver = MediaDriver.launchEmbedded(mediaDriverCtx);
            log.info("Using media driver directory " + mediaDriver.aeronDirectoryName());
        }

        Aeron.Context ctx = new Aeron.Context().publicationConnectionTimeout(-1)
                .availableImageHandler(AeronUtil::printAvailableImage)
                .unavailableImageHandler(AeronUtil::printUnavailableImage)
                .aeronDirectoryName(mediaDriver.aeronDirectoryName())
                .keepAliveInterval(1000)
                .errorHandler(e -> log.error(e.toString(), e));


        if(master) {
            callback =  new ParameterAveragingListener(parameterLength);
            //start an extra daemon for responding to get queries
            ParameterAveragingListener cast = (ParameterAveragingListener) callback;
            responder = AeronNDArrayResponder.startSubscriber(
                    getContext(),
                    host,port + 1,
                    cast,
                    streamId + 1);
            log.info("Started responder on master node " + responder.connectionUrl());
        }
        else {
            String[] publishMasterUrlArr = publishMasterUrl.split(":");
            if(publishMasterUrlArr == null || publishMasterUrlArr.length < 2)
                throw new IllegalStateException("Please specify publish master url as host:port");

            callback = new PublishingListener(
                    String.format("aeron:udp?endpoint=%s:%s",
                            publishMasterUrlArr[0],
                            publishMasterUrlArr[1]),
                    Integer.parseInt(publishMasterUrlArr[2]),
                    getContext());
        }

        log.info("Starting subscriber on " +  host + ":" + port + " and stream " + streamId);
        AtomicBoolean running = new AtomicBoolean(true);

        //start a node
        subscriber = AeronNDArraySubscriber.startSubscriber(
                getContext(),
                host,port,
                callback,
                streamId,running);

        while(!subscriber.launched()) {
            try {
                Thread.sleep(1000);
                log.warn(String.format("Subscriber for channel %s not launched yet...waiting", AeronConnectionInformation.of(host,port,streamId).toString()));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

    }

    //get a context
    public Aeron.Context getContext() {
        Aeron.Context ctx = new Aeron.Context().publicationConnectionTimeout(-1)
                .availableImageHandler(AeronUtil::printAvailableImage)
                .unavailableImageHandler(AeronUtil::printUnavailableImage)
                .aeronDirectoryName(mediaDriver.aeronDirectoryName())
                .keepAliveInterval(10000)
                .errorHandler(e -> log.error(e.toString(), e));
        return ctx;
    }

    /**
     * Returns true if the subscriber is launched
     * @return
     */
    public boolean subscriberLaunched() {
        return subscriber.launched();
    }

    public static void main(String[] args) {
        new ParameterAveragingSubscriber().run(args);
    }
}
