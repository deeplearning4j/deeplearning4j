package org.arbiter.optimize.ui.resources;

import org.arbiter.optimize.ui.components.RenderableComponentString;
import org.arbiter.optimize.ui.rendering.RenderElements;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**Results for each candidate model individually
 */
@Path("/modelResults")
@Produces(MediaType.APPLICATION_JSON)
public class CandidateResultsResource {

    private Map<Integer,RenderElements> map = new ConcurrentHashMap<>();
    private static final RenderElements NOT_FOUND = new RenderElements(new RenderableComponentString("(Not found)"));
    private final Map<Integer,Long> lastUpdateTimeMap = new ConcurrentHashMap<>();

    @GET
    @Path("/{id}")
    public Response getModelStatus(@PathParam("id") int candidateId){
//        if(!map.containsKey(candidateId)) return Response.ok(NOT_FOUND).build();
        return Response.ok(
                new RenderElements(new RenderableComponentString("Renderable elements for candidate " + candidateId + " goes here..."))
        ).build();
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

}
