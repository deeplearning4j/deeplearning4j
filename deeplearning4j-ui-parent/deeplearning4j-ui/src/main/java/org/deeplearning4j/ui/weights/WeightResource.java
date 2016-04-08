/*
 *
 *  * Copyright 2015 Skymind,Inc.
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

package org.deeplearning4j.ui.weights;

import com.codahale.metrics.annotation.ExceptionMetered;
import io.dropwizard.views.View;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.ui.storage.HistoryStorage;
import org.deeplearning4j.ui.storage.SessionStorage;
import org.deeplearning4j.ui.storage.def.ObjectType;
import org.deeplearning4j.ui.weights.beans.CompactModelAndGradient;


import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.Collections;
import java.util.HashMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;


/**
 * Weight renderings
 *
 *
 * @author Adam Gibson
 */
@Path("/weights")
public class WeightResource {
    String path = "weights";
    private SessionStorage storage = SessionStorage.getInstance();

    @GET
    @Path("/updated")
    @Produces(MediaType.APPLICATION_JSON)
    public Response updated(@QueryParam("sid") String sessionId) {
        CompactModelAndGradient model = (CompactModelAndGradient) storage.getObject(sessionId, ObjectType.HISTOGRAM);
        if (model == null) {
            return Response.noContent().build();
        } else return Response.ok(Collections.singletonMap("status",true)).build();
    }

    @GET
    @Path("/data")
    @Produces(MediaType.APPLICATION_JSON)
    public Response data(@QueryParam("sid") String sessionId) {
        //initialized with empty data
        CompactModelAndGradient model = (CompactModelAndGradient) storage.getObject(sessionId, ObjectType.HISTOGRAM);

            if (model == null) {
                //initialize with empty
                return Response.ok(new HashMap<>()).build();
            }

        return Response.ok(model).build();
    }


    @POST
    @Path("/update")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    @ExceptionMetered
    public Response update(CompactModelAndGradient modelAndGrad, @PathParam("path") String path, @QueryParam("sid") String sessionId) {

            storage.putObject(sessionId, ObjectType.HISTOGRAM, modelAndGrad);

            return Response.ok(Collections.singletonMap("status","ok")).build();
    }



}
