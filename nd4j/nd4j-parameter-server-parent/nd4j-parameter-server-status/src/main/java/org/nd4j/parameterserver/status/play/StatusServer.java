/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.parameterserver.status.play;


import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.model.MasterStatus;
import org.nd4j.parameterserver.model.ServerTypeJson;
import org.nd4j.parameterserver.model.SlaveStatus;
import org.nd4j.parameterserver.model.SubscriberState;
import play.BuiltInComponents;
import play.Mode;
import play.libs.Json;
import play.routing.Router;
import play.routing.RoutingDsl;
import play.server.Server;

import static play.libs.Json.toJson;
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
@Slf4j
public class StatusServer {

    /**
     * Start a server based on the given subscriber.
     * Note that for the port to start the server on, you should
     * set the statusServerPortField on the subscriber
     * either manually or via command line. The
     * server defaults to port 9000.
     *
     * The end points are:
     * /opType: returns the opType information (master/slave)
     * /started: if it's a master node, it returns master:started/stopped and responder:started/stopped
     * /connectioninfo: See the SlaveConnectionInfo and MasterConnectionInfo classes for fields.
     * /ids: the list of ids for all of the subscribers
     * @param statusStorage the subscriber to base
     *                   the status server on
     * @return the started server
     */
    public static Server startServer(StatusStorage statusStorage, int statusServerPort) {
        log.info("Starting server on port " + statusServerPort);
        return Server.forRouter(Mode.PROD, statusServerPort, builtInComponents -> createRouter(statusStorage, builtInComponents));
    }

    protected static Router createRouter(StatusStorage statusStorage, BuiltInComponents builtInComponents){
        RoutingDsl dsl = RoutingDsl.fromComponents(builtInComponents);
        dsl.GET("/ids/").routingTo(request -> ok(toJson(statusStorage.ids())));
        dsl.GET("/state/:id").routingTo((request, id) -> ok(toJson(statusStorage.getState(Integer.parseInt(id.toString())))));
        dsl.GET("/opType/:id").routingTo((request, id) -> ok(toJson(ServerTypeJson.builder()
                .type(statusStorage.getState(Integer.parseInt(id.toString())).serverType()))));
        dsl.GET("/started/:id").routingTo((request, id) -> {
            boolean isMaster = statusStorage.getState(Integer.parseInt(id.toString())).isMaster();
            if(isMaster){
                return ok(toJson(MasterStatus.builder().master(statusStorage.getState(Integer.parseInt(id.toString())).getServerState())
                        //note here that a responder is id + 1
                        .responder(statusStorage.getState(Integer.parseInt(id.toString()) + 1).getServerState())
                        .responderN(statusStorage.getState(Integer.parseInt(id.toString())).getTotalUpdates())
                        .build()));
            } else {
                return ok(toJson(SlaveStatus.builder().slave(statusStorage.getState(Integer.parseInt(id.toString())).serverType()).build()));
            }
        });
        dsl.GET("/connectioninfo/:id").routingTo((request, id) -> ok(toJson(statusStorage.getState(Integer.parseInt(id.toString())).getConnectionInfo())));

        dsl.POST("/updatestatus/:id").routingTo((request, id) -> {
            SubscriberState subscriberState = Json.fromJson(request.body().asJson(), SubscriberState.class);
            statusStorage.updateState(subscriberState);
            return ok(toJson(subscriberState));
        });

        return dsl.build();
    }
}
