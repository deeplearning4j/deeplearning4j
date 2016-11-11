package org.nd4j.parameterserver.status.play;

import org.nd4j.parameterserver.model.*;
import org.nd4j.parameterserver.ParameterServerSubscriber;

import play.routing.RoutingDsl;
import play.server.Server;
import static play.mvc.Controller.*;
import static play.libs.Json.*;


/**
 * Play server for communicating
 * the status of
 * the subscriber daemon.
 *
 * This is a rest server for communicating
 * information such as whether the server is started or ont
 * as well as additional connection information.
 *
 * This is mainly meant for internal use.
 *
 * @author Adam Gibson
 */
public class StatusServer {
    /**
     * Start a server based on the given subscriber.
     * Note that for the port to start the server on, you should
     * set the statusServerPortField on the subscriber
     * either manually or via command line. The
     * server defaults to port 9000.
     *
     * The end points are:
     * /type: returns the type information (master/slave)
     * /started: if it's a master node, it returns master:started/stopped and responder:started/stopped
     * /connectioninfo: See the SlaveConnectionInfo and MasterConnectionInfo classes for fields.
     *
     * @param subscriber the subscriber to base
     *                   the status server on
     * @return the started server
     */
    public static Server startServer(final ParameterServerSubscriber subscriber) {
        RoutingDsl dsl = new RoutingDsl();
        dsl.GET("/type").routeTo(() -> ok(toJson(
                ServerTypeJson.builder().type(subscriber.isMaster()
                        ? ServerType.MASTER.name().toLowerCase()
                        : ServerType.SLAVE.name().
                        toLowerCase()).build())));
        dsl .GET("/started").routeTo(() -> subscriber.isMaster() ?
                ok(toJson(MasterStatus.builder()
                        .master(subscriber.subscriberLaunched() ?
                                ServerState.STARTED.name().toLowerCase() :
                                ServerState.STOPPED.name().toLowerCase())
                        .responder(subscriber.getResponder().getLaunched().get() ?
                                ServerState.STARTED.name().toLowerCase() :
                                ServerState.STOPPED.name().toLowerCase())
                        .responderN(subscriber.getResponder().getNdArrayHolder().totalUpdates())
                        .build()))
                :  ok(toJson(SlaveStatus.builder().slave(
                subscriber.subscriberLaunched() ?
                        ServerState.STARTED.name().toLowerCase()
                        : ServerState.STOPPED.name().toLowerCase()).build())));
        dsl.GET("/connectioninfo").routeTo(() -> subscriber.isMaster() ?
                ok(toJson(subscriber.masterConnectionInfo())) :
                ok(toJson(subscriber.slaveConnectionInfo())))
                .build();

        Server server = Server.forRouter(dsl.build(), subscriber.getStatusServerPort());

        return server;

    }


}
