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


import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.Collections;
import java.util.HashMap;


/**
 * Weight renderings
 *
 * @author Adam Gibson
 */
@Path("/weights")
public class WeightResource {
    private ModelAndGradient current;
    String path = "weights";
    private boolean updated = true;
    @GET
    @Produces(MediaType.TEXT_HTML)
    public View get() {
        return new WeightView(path);
    }

    @GET
    @Path("/updated")
    @Produces(MediaType.APPLICATION_JSON)
    public Response updated() {
        return Response.ok(Collections.singletonMap("status",true)).build();
    }

    @GET
    @Path("/data")
    @Produces(MediaType.APPLICATION_JSON)
    public Response data() {
        //initialized with empty data
        if(current == null) {
            //initialize with empty
            updated = false;
            return Response.ok(new HashMap<>()).build();

        }

        //cache response; don't refetch data
        updated = false;
        return Response.ok(current).build();
    }


    @POST
    @Path("/update")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    @ExceptionMetered
    public Response update(ModelAndGradient modelAndGrad, @PathParam("path") String path) {
        this.current = modelAndGrad;
        this.path = modelAndGrad.getPath();
        updated = true;
        return Response.ok(Collections.singletonMap("status","ok")).build();
    }



}
