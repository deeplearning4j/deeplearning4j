package org.nd4j.parameterserver;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import com.beust.jcommander.Parameters;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import com.mashape.unirest.http.Unirest;
import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import io.aeron.driver.ThreadingMode;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.agrona.CloseHelper;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.json.JSONObject;
import org.nd4j.aeron.ipc.AeronNDArraySubscriber;
import org.nd4j.aeron.ipc.AeronUtil;
import org.nd4j.aeron.ipc.NDArrayCallback;
import org.nd4j.aeron.ipc.NDArrayHolder;
import org.nd4j.aeron.ipc.response.AeronNDArrayResponder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.parameterserver.model.MasterConnectionInfo;
import org.nd4j.parameterserver.model.ServerState;
import org.nd4j.parameterserver.model.SlaveConnectionInfo;
import org.nd4j.parameterserver.model.SubscriberState;
import org.nd4j.parameterserver.util.CheckSocket;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.LockSupport;

/**
 * Subscriber main class for
 * the parameter
 * averaging server
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
@Data
@Parameters(separators = ",")
public class ParameterServerSubscriber {

    private static Logger log = LoggerFactory.getLogger(ParameterServerSubscriber.class);

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
    @Parameter(names={"-md","--mediadriverdirectory"}, description = "The media driver directory name. This is for when the media driver is started as a separate process.", arity = 1)
    private String mediaDriverDirectoryName;
    @Parameter(names={"-sp","--statusserverport"}, description = "The status server port, defaults to 9000.", arity = 1)
    private int statusServerPort = 9000;
    @Parameter(names={"-sh","--statusserverhost"}, description = "The status host, defaults to localhost.", arity = 1)
    private String statusServerHost = "localhost";
    @Parameter(names={"-up","--update"}, description = "The update type for this parameter server. Defaults to synchronous", arity = 1)
    private String updateTypeString;

    private UpdateType updateType = UpdateType.SYNC;

    @Parameter(names={"-s","--shape"}, description = "The shape of the ndarray", arity = 1)
    private List<Integer> shape;
    @Parameter(names={"-hbi","--heartbeatinterval"}, description = "The shape of the ndarray", arity = 1)
    private int heartbeatMs = 1000;
    private ObjectMapper objectMapper = new ObjectMapper();
    private ScheduledExecutorService scheduledExecutorService;
    public enum UpdateType {
        HOGWILD,SYNC,TIME_DELAYED,SOFTSYNC
    }


    private MediaDriver mediaDriver;
    private AeronNDArrayResponder responder;
    private AeronNDArraySubscriber subscriber;
    private NDArrayCallback callback;
    //alias for the callback where relevant
    private ParameterServerListener parameterServerListener;
    private Aeron aeron;
    private ScheduledExecutorService heartbeat;

    /**
     * Allow passing in a
     * media driver that already exists
     * @param mediaDriver
     */
    public ParameterServerSubscriber(MediaDriver mediaDriver) {
        Preconditions.checkNotNull(mediaDriver);
        this.mediaDriver = mediaDriver;
    }


    /**
     * Return the current {@link SubscriberState}
     * of this subscriber
     * @return the current state of this subscriber
     */
    public SubscriberState asState() {
        return SubscriberState.builder()
                .parameterUpdaterStatus(parameterServerListener == null ? Collections.emptyMap() : parameterServerListener.getUpdater().status())
                .isMaster(isMaster()).connectionInfo(isMaster() ? slaveConnectionInfo().toString() : masterConnectionInfo().toString())
                .totalUpdates(getResponder().getNdArrayHolder().totalUpdates())
                .serverState(subscriberLaunched() ?
                        ServerState.STARTED.name().toLowerCase() :
                        ServerState.STOPPED.name().toLowerCase())
                .build();
    }

    /**
     * When this is a slave node
     * it returns the connection url for this node
     * and the associated master connection urls in the form of:
     * host:port:streamId
     * @return the slave connection info
     */
    public SlaveConnectionInfo slaveConnectionInfo() {
        if(isMaster())
            throw new IllegalStateException("Unable to determine slave connection info. This is a master node");
        return SlaveConnectionInfo.builder()
                .connectionUrl(subscriber.connectionUrl())
                .masterUrl(publishMasterUrl).build();

    }


    /**
     * When this is a master node,
     * it returns the connection url for this node,
     * it's slaves (if any exist) and the responder
     * connection url in the form of:
     * host:port:streamId
     * @return the master connection info
     */
    public MasterConnectionInfo masterConnectionInfo() {
        if(!isMaster())
            throw new IllegalStateException("Unable to determine master connection info. This is a slave node");
        return MasterConnectionInfo.builder()
                .connectionUrl(subscriber.connectionUrl())
                .responderUrl(responder.connectionUrl())
                .slaveUrls(new ArrayList<>()).build();
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
            System.exit(1);
        }

        if(publishMasterUrl == null && !master)
            throw new IllegalStateException("Please specify a master url or set master to true");

        //allows passing in a media driver for things like unit tests
        //also ensure we don't use a media driver when a directory is specified
        //for a remote one
        if(mediaDriver == null && mediaDriverDirectoryName == null) {
            //length of array * sizeof(float)
            int ipcLength = ArrayUtil.prod(Ints.toArray(shape)) * 4;
            //must be a power of 2
            ipcLength *= 2;
            //padding for NDArrayMessage
            ipcLength += 64;
            //Length in bytes for the SO_RCVBUF, 0 means use OS default. This needs to be larger than Receiver Window.
            System.setProperty("aeron.socket.so_rcvbuf",String.valueOf(ipcLength));
            final MediaDriver.Context mediaDriverCtx = new MediaDriver.Context()
                    .threadingMode(ThreadingMode.DEDICATED)
                    .dirsDeleteOnStart(deleteDirectoryOnStart)
                    .termBufferSparseFile(false)
                    .ipcTermBufferLength(ipcLength)
                    .publicationTermBufferLength(ipcLength)
                    .maxTermBufferLength(ipcLength)
                    .conductorIdleStrategy(new BusySpinIdleStrategy())
                    .receiverIdleStrategy(new BusySpinIdleStrategy())
                    .senderIdleStrategy(new BusySpinIdleStrategy());

            mediaDriver = MediaDriver.launchEmbedded(mediaDriverCtx);
            //set the variable since we are using a media driver directly
            mediaDriverDirectoryName = mediaDriver.aeronDirectoryName();
            log.info("Using media driver directory " + mediaDriver.aeronDirectoryName());
        }

        if(aeron == null)
            this.aeron = Aeron.connect(getContext());



        if(master) {
            //instantiate with shape instead of just length
            callback =  new ParameterServerListener(Ints.toArray(shape));
            parameterServerListener = (ParameterServerListener) callback;
            //start an extra daemon for responding to get queries
            ParameterServerListener cast = (ParameterServerListener) callback;
            responder = AeronNDArrayResponder.startSubscriber(
                    aeron,
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
                aeron,
                host,port,
                callback,
                streamId,running);

        while(!subscriber.launched()) {
            LockSupport.parkNanos(100000);
        }

        //send heartbeat to a status server. There will usually be 1 status server per master.
        //Only schedule this if a remote server is available.
       if(CheckSocket.remotePortTaken(statusServerHost,statusServerPort,10000)) {
           scheduledExecutorService = Executors.newScheduledThreadPool(1);

           scheduledExecutorService.scheduleAtFixedRate(() -> {
               try {
                   JSONObject jsonObject = new JSONObject(objectMapper.writeValueAsString(asState()));
                   Unirest.post(String.format("http://%s:%d/updatestatus/%d",statusServerHost,statusServerPort,streamId))
                           .body(jsonObject).getEntity();
               } catch (JsonProcessingException e) {
                   e.printStackTrace();
               }
           },1000,heartbeatMs, TimeUnit.MILLISECONDS);

       }
       else {
            log.info("No status server found. Will not send heartbeats. Specified host was " + statusServerHost + " and port was " + statusServerPort);
       }


        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if(subscriber != null)
                CloseHelper.quietClose(subscriber);
            if(responder != null)
                CloseHelper.quietClose(responder);
            if(aeron != null)
                CloseHelper.quietClose(aeron);
            if(scheduledExecutorService != null)
                scheduledExecutorService.shutdown();

        }));

        //set the server for the status of the master and slave nodes



    }

    //get a context
    public Aeron.Context getContext() {
        Aeron.Context ctx = new Aeron.Context().publicationConnectionTimeout(-1)
                .availableImageHandler(AeronUtil::printAvailableImage)
                .unavailableImageHandler(AeronUtil::printUnavailableImage)
                .aeronDirectoryName(mediaDriverDirectoryName)
                .keepAliveInterval(100000)
                .errorHandler(e -> log.error(e.toString(), e));
        return ctx;
    }

    /**
     * Get the master ndarray from the
     * internal {@link NDArrayHolder}
     * @return the master ndarray
     */
    public INDArray getMasterArray() {
        NDArrayHolder holder = (NDArrayHolder) callback;
        return holder.get();
    }


    /**
     * Returns true if the subscriber is launched
     * @return
     */
    public boolean subscriberLaunched() {
        return subscriber.launched();
    }

    public static void main(String[] args) {
        new ParameterServerSubscriber().run(args);
    }
}
