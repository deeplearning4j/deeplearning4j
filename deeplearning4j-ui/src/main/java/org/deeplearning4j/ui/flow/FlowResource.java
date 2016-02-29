package org.deeplearning4j.ui.flow;

import org.deeplearning4j.ui.flow.beans.ModelInfo;
import org.deeplearning4j.ui.flow.beans.NodeReport;
import org.deeplearning4j.ui.storage.SessionStorage;
import org.deeplearning4j.ui.storage.def.ObjectType;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.Collections;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Almost RESTful interface for FlowIterationListener.
 *
 * @author raver119@gmail.com
 */
@Path("/flow")
public class FlowResource {
    private SessionStorage storage = SessionStorage.getInstance();

    @GET
    @Produces(MediaType.TEXT_HTML)
    public FlowView getView() {
        return new FlowView();
    }

    @GET
    @Path("/state")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getState(@QueryParam("sid") String sessionId) {
        // TODO: to be improved with HistoryStorage
        ModelInfo model = (ModelInfo) storage.getObject(sessionId, ObjectType.FLOW);
        return Response.ok(model).build();
    }

    @POST
    @Path("/state")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response postState(ModelInfo info, @QueryParam("sid") String sessionId) {
        storage.putObject(sessionId, ObjectType.FLOW, info);

        return Response.ok(Collections.singletonMap("status","ok")).build();
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
