package org.deeplearning4j.ui.module.flow;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.ui.api.FunctionType;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.flow.data.FlowStaticPersistable;
import org.deeplearning4j.ui.flow.data.FlowUpdatePersistable;
import play.libs.Json;
import play.mvc.Result;

import java.util.*;

import static play.mvc.Results.ok;

/**
 * Module for FlowIterationListener
 *
 * @author Alex Black
 */
@Slf4j
public class FlowListenerModule implements UIModule {

    private static final String TYPE_ID = "FlowListener";

    private Map<String, StatsStorage> knownSessionIDs = Collections.synchronizedMap(new LinkedHashMap<>());

    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.singletonList(TYPE_ID);
    }

    @Override
    public List<Route> getRoutes() {
        Route r1 = new Route("/flow", HttpMethod.GET, FunctionType.Supplier,
                        () -> ok(org.deeplearning4j.ui.views.html.flow.Flow.apply()));
        Route r2 = new Route("/flow/info/:id", HttpMethod.GET, FunctionType.Function, this::getStaticInfo);
        Route r3 = new Route("/flow/state/:id", HttpMethod.GET, FunctionType.Function, this::getUpdate);
        Route r4 = new Route("/flow/listSessions", HttpMethod.GET, FunctionType.Supplier, this::listSessions);

        return Arrays.asList(r1, r2, r3, r4);
    }

    @Override
    public void reportStorageEvents(Collection<StatsStorageEvent> events) {
        //We should only be getting relevant session IDs...
        for (StatsStorageEvent sse : events) {
            if (!knownSessionIDs.containsKey(sse.getSessionID())) {
                knownSessionIDs.put(sse.getSessionID(), sse.getStatsStorage());
            }
        }
    }

    @Override
    public void onAttach(StatsStorage statsStorage) {
        for (String sessionID : statsStorage.listSessionIDs()) {
            for (String typeID : statsStorage.listTypeIDsForSession(sessionID)) {
                if (!TYPE_ID.equals(typeID))
                    continue;
                knownSessionIDs.put(sessionID, statsStorage);
            }
        }
    }

    @Override
    public void onDetach(StatsStorage statsStorage) {
        for (String s : knownSessionIDs.keySet()) {
            if (statsStorage == knownSessionIDs.get(s)) {
                knownSessionIDs.remove(s);
            }
        }
    }

    private Result listSessions() {
        return ok(Json.toJson(knownSessionIDs.keySet()));
    }

    private Result getStaticInfo(String sessionID) {
        if (!knownSessionIDs.containsKey(sessionID))
            return ok("Unknown session ID");
        StatsStorage ss = knownSessionIDs.get(sessionID);

        List<Persistable> list = ss.getAllStaticInfos(sessionID, TYPE_ID);
        if (list == null || list.size() == 0)
            return ok();

        Persistable p = list.get(0);
        if (!(p instanceof FlowStaticPersistable))
            return ok();

        FlowStaticPersistable f = (FlowStaticPersistable) p;

        return ok(Json.toJson(f.getModelInfo()));
    }

    private Result getUpdate(String sessionID) {
        if (!knownSessionIDs.containsKey(sessionID))
            return ok("Unknown session ID");
        StatsStorage ss = knownSessionIDs.get(sessionID);

        List<Persistable> list = ss.getLatestUpdateAllWorkers(sessionID, TYPE_ID);
        if (list == null || list.size() == 0)
            return ok();

        Persistable p = list.get(0);
        if (!(p instanceof FlowUpdatePersistable))
            return ok();

        FlowUpdatePersistable f = (FlowUpdatePersistable) p;

        return ok(Json.toJson(f.getModelState()));
    }
}
