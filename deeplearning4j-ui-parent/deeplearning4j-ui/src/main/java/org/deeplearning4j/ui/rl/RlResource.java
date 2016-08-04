package org.deeplearning4j.ui.rl;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.ui.rl.beans.ReportBean;
import org.deeplearning4j.ui.storage.HistoryStorage;

import javax.ws.rs.*;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Almost RESTful interface for FlowIterationListener.
 *
 * @author raver119@gmail.com
 */
@Path("/rl")
public class RlResource {
    //private SessionStorage storage = SessionStorage.getInstance();
    private HistoryStorage storage = HistoryStorage.getInstance();
    private String key = "RL";

    @GET
    @Path("/state")
    @Produces(MediaType.APPLICATION_JSON)
    public Response getState(@QueryParam("sid") String sessionId) {

        // FIXME: getSorted should use derived types!

        List<Object> rewards = storage.getSorted(key, HistoryStorage.SortOutput.ASCENDING);
        List<String> conv = new ArrayList<>();

        for (Object object: rewards) {
            ReportBean bean = (ReportBean) object;
            conv.add(new String("" + bean.getEpochId() + "|" + bean.getReward() + "|Epoch_" + bean.getEpochId()));
        }

        return Response.ok(conv).build();
    }

    @POST
    @Path("/state")
    @Consumes(MediaType.APPLICATION_JSON)
    @Produces(MediaType.APPLICATION_JSON)
    public Response postState(@QueryParam("sid") String sessionId, ReportBean bean) {

        storage.put(key, Pair.newPair((int) bean.getEpochId(), 0), bean);

        return Response.ok(Collections.singletonMap("status", "ok")).build();
    }
}

