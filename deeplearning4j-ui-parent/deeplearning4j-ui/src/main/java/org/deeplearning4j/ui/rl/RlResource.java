package org.deeplearning4j.ui.rl;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.ui.flow.beans.ModelInfo;
import org.deeplearning4j.ui.flow.beans.NodeReport;
import org.deeplearning4j.ui.rl.beans.ReportBean;
import org.deeplearning4j.ui.storage.HistoryStorage;
import org.deeplearning4j.ui.storage.SessionStorage;
import org.deeplearning4j.ui.storage.def.ObjectType;

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
        List<ReportBean> conv = new ArrayList<>();
        for (Object object: rewards) {
            conv.add((ReportBean) object);
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

