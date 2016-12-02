package org.nd4j.parameterserver.status.play;

import com.fasterxml.jackson.databind.JsonNode;
import org.nd4j.parameterserver.ParameterServerSubscriber;
import org.nd4j.parameterserver.model.*;
import play.libs.F;
import play.libs.Json;
import play.mvc.Result;
import play.routing.RoutingDsl;
import play.server.Server;

import java.util.List;

import static play.libs.Json.*;
import static play.libs.Json.toJson;
import static play.mvc.Controller.*;
import static play.mvc.Results.ok;


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
     * /ids: the list of ids for all of the subscribers
     * @param statusStorage the subscriber to base
     *                   the status server on
     * @return the started server
     */
    public static Server startServer(StatusStorage statusStorage,int statusServerPort) {
        RoutingDsl dsl = new RoutingDsl();
        dsl.GET("/ids").routeTo(new F.Function<Object,Result>() {

            @Override
            public Result apply(Object o) throws Throwable {
                List<Integer> ids = statusStorage.ids();
                return ok(toJson(ids));
            }
        });
        dsl.GET("/type/:id").routeTo(new F.Function<String, Result>() {
            @Override
            public Result apply(String id) throws Throwable {
                return ok(toJson(
                        ServerTypeJson.builder().type(statusStorage.getState(Integer.parseInt(id)).serverType())));
            }
        });


        dsl .GET("/started/:id").routeTo(new F.Function<String, Result>() {
            @Override
            public Result apply(String id) throws Throwable {
                return statusStorage.getState(Integer.parseInt(id)).isMaster() ?
                        ok(toJson(MasterStatus.builder()
                                .master(statusStorage.getState(Integer.parseInt(id)).getServerState())
                                //note here that a responder is is + 1
                                .responder(statusStorage.getState(Integer.parseInt(id) + 1).getServerState())
                                .responderN(statusStorage.getState(Integer.parseInt(id)).getTotalUpdates())
                                .build()))
                        :  ok(toJson(SlaveStatus.builder().slave(
                        statusStorage.getState(Integer.parseInt(id)).serverType()).build()));
            }
        });



        dsl.GET("/connectioninfo/:id").routeTo(new F.Function<String, Result>() {
            @Override
            public Result apply(String id) throws Throwable {
                return ok(toJson(statusStorage.getState(Integer.parseInt(id)).getConnectionInfo()));
            }
        });
        dsl.POST("/updatestatus/:id").routeTo(new F.Function<String, Result>() {
            @Override
            public Result apply(String id) throws Throwable {
                SubscriberState subscriberState = Json.fromJson(request().body().asJson(),SubscriberState.class);
                statusStorage.updateState(subscriberState);
                return ok(toJson(subscriberState));
            }
        });

        Server server = Server.forRouter(dsl.build(), statusServerPort);

        return server;

    }


}
