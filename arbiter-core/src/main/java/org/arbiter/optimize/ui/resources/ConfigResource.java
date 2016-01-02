/*
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
package org.arbiter.optimize.ui.resources;

import org.arbiter.optimize.ui.components.RenderElements;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.Collections;

@Path("/config")
@Produces(MediaType.APPLICATION_JSON)
public class ConfigResource {

    public static final Logger log = LoggerFactory.getLogger(ConfigResource.class);
    private RenderElements renderElements = new RenderElements();

    @GET
    public Response getConfig(){
        log.info("GET for config with current config: {}");
        return Response.ok(renderElements).build();
    }

    @POST
    @Path("/update")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response update(RenderElements renderElements){
        log.info("POST for config with new elements: {}",renderElements);
        this.renderElements = renderElements;
        return Response.ok(Collections.singletonMap("status", "ok")).build();
    }

}
