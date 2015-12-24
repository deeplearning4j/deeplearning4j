package org.arbiter.optimize.ui.resources;

import org.arbiter.optimize.runner.CandidateStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Results for the results table (non-expanded)*/
@Path("/results")
@Produces(MediaType.APPLICATION_JSON)
public class SummaryResultsResource {

    public static final Logger log = LoggerFactory.getLogger(SummaryResultsResource.class);
    private List<CandidateStatus> statusList = new ArrayList<>();

    @GET
    public Response getCandidateStatus(){
        log.info("GET for candidate status with current status: {}",statusList);
        return Response.ok(statusList).build();
    }

    @POST
    @Path("/update")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response update(List<CandidateStatus> statusList){
        log.info("POST for results with new status: {}",statusList);
        this.statusList = statusList;
        return Response.ok(Collections.singletonMap("status","ok")).build();
    }

}
