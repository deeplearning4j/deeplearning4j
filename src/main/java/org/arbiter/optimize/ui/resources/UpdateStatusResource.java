package org.arbiter.optimize.ui.resources;

import org.arbiter.optimize.ui.UpdateStatus;

import javax.ws.rs.Consumes;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.Collections;

public class UpdateStatusResource {

    private UpdateStatus status;

    @POST
    @Path("/updateStatus")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response update(UpdateStatus updateStatus){
        this.status = updateStatus;
        return Response.ok(Collections.singletonMap("status", "ok")).build();
    }

}
