package org.arbiter.optimize.ui.resources;

import org.arbiter.optimize.ui.UpdateStatus;
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
        log.info("GET with update status: {}", status);
        return Response.ok(status).build();
    }

    @POST
    @Path("/update")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response update(UpdateStatus updateStatus){
        log.info("POST with last update status: {}", updateStatus);
        this.status = updateStatus;
        return Response.ok(Collections.singletonMap("status", "ok")).build();
    }

}
