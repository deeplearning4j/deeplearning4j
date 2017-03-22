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
import org.deeplearning4j.ui.components.text.ComponentText;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**Results for each candidate model individually
 */
@Path("/modelResults")
@Produces(MediaType.APPLICATION_JSON)
public class CandidateResultsResource {

    private static final Logger log = LoggerFactory.getLogger(CandidateResultsResource.class);

    private static final int maxWarnCount = 10;
    private AtomicInteger warnCount = new AtomicInteger(0);

    private Map<Integer,Component> map = new ConcurrentHashMap<>();
    private static final Component NOT_FOUND = new ComponentText("(Candidate results: Not found)",null);
    private final Map<Integer,Long> lastUpdateTimeMap = new ConcurrentHashMap<>();

    @GET
    @Path("/{id}")
    public Response getModelStatus(@PathParam("id") int candidateId){
        if(!map.containsKey(candidateId)) return Response.ok(NOT_FOUND).build();
        String str = "";
        try{
            Component c = map.get(candidateId);
            str = JsonMapper.getMapper().writeValueAsString(c);
        } catch (Exception e){
            if(warnCount.getAndIncrement() < maxWarnCount){
                log.warn("Error getting candidate info", e);
            }
        }
        return Response.ok(str).build();
    }

    @GET
    @Path("/lastUpdate")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response getLastUpdateTimes(@QueryParam("id")List<String> modelIDs){
        //Here: requests are of the form /modelResults/lastUpdate?id=0&id=1&id=2
        Map<Integer,Long> outMap = new HashMap<>();
        for( String id : modelIDs ){
            if(!lastUpdateTimeMap.containsKey(id)) outMap.put(Integer.valueOf(id),0L);
            else outMap.put(Integer.valueOf(id),lastUpdateTimeMap.get(id));
        }
        return Response.ok(outMap).build();
    }

    @POST
    @Path("/update/{id}")
    @Consumes(MediaType.TEXT_PLAIN)
    @Produces(MediaType.APPLICATION_JSON)
    public Response update(@PathParam("id")int candidateID, String componentStr){

        if(componentStr == null || componentStr.isEmpty()){
            return Response.ok(Collections.singletonMap("status", "ok")).build();
        }

        try{
            Component component = JsonMapper.getMapper().readValue(componentStr, Component.class);
            map.put(candidateID,component);
        } catch (Exception e) {
            if(warnCount.getAndIncrement() < maxWarnCount){
                log.warn("Error posting summary status update", e);
            }
        }

        return Response.ok(Collections.singletonMap("status", "ok")).build();
    }

}
