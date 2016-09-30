package org.nd4j.parameterserver.parameteraveraging;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import io.aeron.driver.ThreadingMode;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.nd4j.aeron.ipc.AeronNDArraySubscriber;
import org.nd4j.aeron.ipc.AeronUtil;
import org.nd4j.aeron.ipc.NDArrayCallback;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Subscriber main class for
 * the parameter
 * averaging server
 *
 * @author Adam Gibson
 */
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
    /**
     *
     * @param args
     */
    public void run(String[] args) {
        JCommander jcmdr = new JCommander(this);
        try{
            jcmdr.parse(args);
        } catch(ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try{ Thread.sleep(500); } catch(Exception e2){ }
            throw e;
        }

        final MediaDriver.Context mediaDriverCtx = new MediaDriver.Context()
                .threadingMode(ThreadingMode.DEDICATED)
                .dirsDeleteOnStart(deleteDirectoryOnStart)
                .termBufferSparseFile(false)
                .conductorIdleStrategy(new BusySpinIdleStrategy())
                .receiverIdleStrategy(new BusySpinIdleStrategy())
                .senderIdleStrategy(new BusySpinIdleStrategy());

        MediaDriver mediaDriver = MediaDriver.launchEmbedded(mediaDriverCtx);
        log.info("Using media driver directory " + mediaDriver.aeronDirectoryName());

        Aeron.Context ctx = new Aeron.Context().publicationConnectionTimeout(-1)
                .availableImageHandler(AeronUtil::printAvailableImage)
                .unavailableImageHandler(AeronUtil::printUnavailableImage)
                .aeronDirectoryName(mediaDriverCtx.aeronDirectoryName())
                .keepAliveInterval(1000)
                .errorHandler(e -> log.error(e.toString(), e));
        NDArrayCallback callback;
        if(master) {
            callback =  new ParameterAveragingListener(parameterLength);;
        }
        else {
            callback = new PublishingListener(String.format("aeron:udp?endpoint=%s:%d",host,port),streamId,ctx);
        }

        //start a node
        AeronNDArraySubscriber.startSubscriber(
                ctx,
                host,port,
                callback,
                streamId);
    }

    public static void main(String[] args) {
        new ParameterAveragingSubscriber().run(args);
    }
}
