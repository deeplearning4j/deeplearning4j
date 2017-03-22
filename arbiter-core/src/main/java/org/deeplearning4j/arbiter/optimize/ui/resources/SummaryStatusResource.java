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

import org.deeplearning4j.arbiter.optimize.serde.jackson.JsonMapper;
import org.deeplearning4j.ui.api.Component;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.Collections;
import java.util.concurrent.atomic.AtomicInteger;

/**Summary stats: number of completed tasks etc
 */
@Path("/summary")
@Produces(MediaType.APPLICATION_JSON)
public class SummaryStatusResource {
    public static Logger log = LoggerFactory.getLogger(SummaryStatusResource.class);

    private static final int maxWarnCount = 5;
    private AtomicInteger warnCount = new AtomicInteger(0);

    private Component component = null;

    @GET
    public Response getStatus(){
        log.trace("Get with elements: {}",component);
        String str = "";
        try{
            str = JsonMapper.getMapper().writeValueAsString(component);
        } catch (Exception e){
            if(warnCount.getAndIncrement() < maxWarnCount){
                log.warn("Error getting summary status update", e);
            }
        }
        Response r = Response.ok(str).build();
        return r;
    }

    @POST
    @Path("/update")
    @Produces(MediaType.APPLICATION_JSON)
    @Consumes(MediaType.TEXT_PLAIN)
    public Response update(String component){
        log.trace("Post with new elements: {}",component);

        if(component == null || component.isEmpty()){
            return Response.ok(Collections.singletonMap("status", "ok")).build();
        }

        try{
            this.component = JsonMapper.getMapper().readValue(component, Component.class);
        } catch (Exception e) {
            if(warnCount.getAndIncrement() < maxWarnCount){
                log.warn("Error posting summary status update", e);
            }
        }

        return Response.ok(Collections.singletonMap("status", "ok")).build();
    }

}
