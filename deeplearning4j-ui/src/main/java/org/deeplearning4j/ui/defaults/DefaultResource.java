package org.deeplearning4j.ui.defaults;

import org.deeplearning4j.ui.storage.SessionStorage;
import org.deeplearning4j.ui.storage.def.ObjectType;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

/**
 * Container for the default root page, that's available for user when he open his browser and navigate to dl4j-ui
 *
 * @author raver119@gmail.com
 */
@Path("/")
@Produces(MediaType.TEXT_HTML)
public class DefaultResource {
    /**
     * So, basic idea here: main page, with links to available activities within this UIServer
     *
     */
    private SessionStorage storage = SessionStorage.getInstance();


    /**
     * This method produces default page, aka index page
     *
     * @return
     */
    @GET
    @Produces(MediaType.TEXT_HTML)
    public DefaultView get() {
        return new DefaultView();
    }

    @GET
    @Path("/whatsup")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getWhatsUp() {
        /*
         TODO: this method should return the list of available processes reported to this UIServer. And this response will be used on client-side
         Actual mechanism, probably, will use HistoryStorage, and it's ability to track active objects within
          */
        return Response.ok(storage.getSessions()).build();
    }

    @GET
    @Path("/events")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getEvents() {
        return Response.ok(storage.getEvents()).build();
    }

    @GET
    @Path("/sessions")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response getSessions(@QueryParam("event") String event) {
        ObjectType type = ObjectType.valueOf(event);
        if (type == null) return Response.noContent().build();

        return Response.ok(storage.getSessions(type)).build();
    }
}
