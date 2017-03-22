/*-
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.deeplearning4j.arbiter.optimize.ui.resources;

import org.deeplearning4j.arbiter.optimize.ui.UpdateStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.Collections;

/** Update status: used to record when updates to each type of info were last posted. */
@Path("/lastUpdate")
@Produces(MediaType.APPLICATION_JSON)
public class LastUpdateResource {
    public static final Logger log = LoggerFactory.getLogger(LastUpdateResource.class);
    private UpdateStatus status = new UpdateStatus(0,0,0);

    @GET
    public Response getStatus(){
        log.trace("GET with update status: {}", status);
        return Response.ok(status).build();
    }

    @POST
    @Path("/update")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response update(UpdateStatus updateStatus){
        log.trace("POST with last update status: {}", updateStatus);
        this.status = updateStatus;
        return Response.ok(Collections.singletonMap("status", "ok")).build();
    }

}
