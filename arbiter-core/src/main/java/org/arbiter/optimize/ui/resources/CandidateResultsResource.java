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

import org.arbiter.optimize.ui.components.RenderableComponentString;
import org.arbiter.optimize.ui.components.RenderElements;
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

/**Results for each candidate model individually
 */
@Path("/modelResults")
@Produces(MediaType.APPLICATION_JSON)
public class CandidateResultsResource {

    private static final Logger log = LoggerFactory.getLogger(CandidateResultsResource.class);

    private Map<Integer,RenderElements> map = new ConcurrentHashMap<>();
    private static final RenderElements NOT_FOUND = new RenderElements(new RenderableComponentString("(Candidate results: Not found)"));
    private final Map<Integer,Long> lastUpdateTimeMap = new ConcurrentHashMap<>();

    @GET
    @Path("/{id}")
    public Response getModelStatus(@PathParam("id") int candidateId){
        if(!map.containsKey(candidateId)) return Response.ok(NOT_FOUND).build();
        return Response.ok(map.get(candidateId)).build();
    }

    @GET
    @Path("/lastUpdate")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response getLastUpdateTimes(@QueryParam("id")List<String> modelIDs){
        //Here: requests are of the form /modelResults/lastUpdate?id=0&id=1&id=2
        //System.out.println("***** Recieved request with IDs: "+modelIDs + " with " + modelIDs.size() + " elements");
        Map<Integer,Long> outMap = new HashMap<>();
        for( String id : modelIDs ){
            if(!lastUpdateTimeMap.containsKey(id)) outMap.put(Integer.valueOf(id),0L);
            else outMap.put(Integer.valueOf(id),lastUpdateTimeMap.get(id));
        }
        return Response.ok(outMap).build();
    }

    @POST
    @Path("/update/{id}")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response update(@PathParam("id")int candidateID, RenderElements renderElements){
//        log.info("Candidate status updated: {}, {}",candidateID,renderElements);
        map.put(candidateID,renderElements);
        return Response.ok(Collections.singletonMap("status", "ok")).build();
    }

}
