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

import io.dropwizard.views.View;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.ui.tsne.TsneView;

import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Weight renderings
 *
 * @author Adam Gibson
 */
@Path("/weights")
@Produces(MediaType.TEXT_HTML)
public class WeightResource {
    private ModelAndGradient current;
    private boolean updated = false;
    @GET
    public View get() {
        return new WeightView();
    }

    @GET
    @Path("/updated")
    @Produces(MediaType.APPLICATION_JSON)
    public Response updated() {
        return Response.ok(Collections.singletonMap("message",updated)).build();
    }

    @GET
    @Path("/data")
    @Produces(MediaType.APPLICATION_JSON)
    public Response data() {
        if(current == null) {
            current = new ModelAndGradient();
            updated = true;
        }
        updated = false;
        return Response.ok(current).build();
    }


    @POST
    @Path("/update")
    @Produces(MediaType.APPLICATION_JSON)
    public Response update(ModelAndGradient modelAndGrad) {
        this.current = modelAndGrad;
        updated = true;
        return Response.ok(Collections.singletonMap("status","ok")).build();
    }




}
