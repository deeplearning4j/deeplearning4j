package org.deeplearning4j.ui.flow;

import org.deeplearning4j.ui.flow.beans.NodeReport;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

/**
 * Almost RESTful interface for FlowIterationListener.
 *
 * @author raver119@gmail.com
 */
@Path("/flow")
public class FlowResource {


    @GET
    @Produces(MediaType.TEXT_HTML)
    public FlowView getView() {
        return new FlowView();
    }

    @GET
    @Path("/action/{id}")
    @Produces(MediaType.TEXT_HTML)
    public Response getAction(@PathParam("id") long id) {
        // TODO: to be implemented
        // TODO: not sure, if we need this to be TEXT_HTML though. Investigate options.
        return Response.ok().build();
    }

    @POST
    @Path("/action/{id}")
    @Consumes(MediaType.APPLICATION_JSON)
    public Response postState(@PathParam("id") long id, NodeReport report) {
        // TODO: to be implemented
        return Response.ok().build();
    }
}
