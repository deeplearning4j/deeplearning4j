package org.deeplearning4j.ui.module.training;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.api.storage.StatsStorageListener;
import org.deeplearning4j.ui.api.*;
import org.deeplearning4j.ui.i18n.I18NProvider;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.stats.api.StatsInitializationReport;
import org.deeplearning4j.ui.stats.api.StatsReport;
import org.deeplearning4j.ui.views.html.training.*;
import play.libs.Json;
import play.mvc.Result;
import play.mvc.Results;

import java.util.*;

import static play.mvc.Results.ok;

/**
 * Created by Alex on 14/10/2016.
 */
@Slf4j
public class TrainModule implements UIModule {

    private Map<String, StatsStorage> knownSessionIDs = new LinkedHashMap<>();
    private String currentSessionID;

    private static final String[][] EMPTY_TABLE = new String[2][0];

//    private static final JsonNode NO_DATA;
//    static {
//        Map<String,Object> noDataMap = new HashMap<>();
//        noDataMap.put("lastScore",0.0);
//        noDataMap.put("scores",Collections.EMPTY_LIST);
//        noDataMap.put("scoresIterCount",Collections.EMPTY_LIST);
//        noDataMap.put("performanceTable", new String[2][0]);
//        NO_DATA = Json.toJson(noDataMap);
//    }

    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.singletonList(StatsListener.TYPE_ID);
    }

    @Override
    public List<Route> getRoutes() {
        Route r = new Route("/train", HttpMethod.GET, FunctionType.Supplier, () -> ok(Training.apply(I18NProvider.getInstance())));
        Route r2 = new Route("/train/overview", HttpMethod.GET, FunctionType.Supplier, () -> ok(TrainingOverview.apply(I18NProvider.getInstance())));
        Route r2a = new Route("/train/overview/data", HttpMethod.GET, FunctionType.Supplier, this::getOverviewData);
        Route r3 = new Route("/train/model", HttpMethod.GET, FunctionType.Supplier, () -> ok(TrainingModel.apply(I18NProvider.getInstance())));
        Route r4 = new Route("/train/system", HttpMethod.GET, FunctionType.Supplier, () -> ok(TrainingSystem.apply(I18NProvider.getInstance())));
        Route r5 = new Route("/train/help", HttpMethod.GET, FunctionType.Supplier, () -> ok(TrainingHelp.apply(I18NProvider.getInstance())));
        Route r6 = new Route("/train/currentSessionID", HttpMethod.GET, FunctionType.Supplier, () -> ok(currentSessionID == null ? "" : currentSessionID));

        return Arrays.asList(r, r2, r2a, r3, r4, r5, r6);
    }

    @Override
    public void reportStorageEvents(StatsStorage statsStorage, Collection<StatsStorageEvent> events) {
        for (StatsStorageEvent sse : events) {
            if (sse.getEventType() == StatsStorageListener.EventType.PostStaticInfo && StatsListener.TYPE_ID.equals(sse.getTypeID())) {
                knownSessionIDs.put(sse.getSessionID(), statsStorage);
            }
        }

        if (currentSessionID == null) getDefaultSession();
    }

    @Override
    public synchronized void onAttach(StatsStorage statsStorage) {
        for (String sessionID : statsStorage.listSessionIDs()) {
            for (String typeID : statsStorage.listTypeIDsForSession(sessionID)) {
                if (!StatsListener.TYPE_ID.equals(typeID)) continue;
                knownSessionIDs.put(sessionID, statsStorage);
            }
        }

        if (currentSessionID == null) getDefaultSession();
    }

    @Override
    public void onDetach(StatsStorage statsStorage) {

    }

    private void getDefaultSession() {
        if (currentSessionID != null) return;

        //TODO handle multiple workers, etc
        long mostRecentTime = Long.MIN_VALUE;
        String sessionID = null;
        for (Map.Entry<String, StatsStorage> entry : knownSessionIDs.entrySet()) {
            List<Persistable> staticInfos = entry.getValue().getAllStaticInfos(entry.getKey(), StatsListener.TYPE_ID);
            if (staticInfos == null || staticInfos.size() == 0) continue;
            Persistable p = staticInfos.get(0);
            long thisTime = p.getTimeStamp();
            if (thisTime > mostRecentTime) {
                mostRecentTime = thisTime;
                sessionID = entry.getKey();
            }
        }

        if (sessionID != null) {
            currentSessionID = sessionID;
        }
    }

    private Result getOverviewData() {
        I18N i18N = I18NProvider.getInstance();

        boolean noData = currentSessionID == null;
        //First pass (optimize later): query all data...

        StatsStorage ss = (noData ? null : knownSessionIDs.get(currentSessionID));


        //TODO HANDLE MULTIPLE WORKERS (SPARK)
        String wid = null;
        if(!noData){
            List<String> workerIDs = ss.listWorkerIDsForSession(currentSessionID);
            if (workerIDs == null || workerIDs.size() == 0) noData = true;
            else {
                wid = workerIDs.get(0);
            }
        }

        List<Integer> scoresIterCount = new ArrayList<>();
        List<Double> scores = new ArrayList<>();

        Map<String, Object> result = new HashMap<>();
        result.put("scores", scores);
        result.put("scoresIter", scoresIterCount);

        //Get scores info
        List<Persistable> updates = (noData ? null : ss.getAllUpdatesAfter(currentSessionID, StatsListener.TYPE_ID, wid, 0));

        StatsReport last = null;
        if(!noData) {
            double lastScore;
            for (Persistable u : updates) {
                if (!(u instanceof StatsReport)) continue;
                last = (StatsReport) u;
                int iterCount = last.getIterationCount();
                scoresIterCount.add(iterCount);
                lastScore = last.getScore();
                scores.add(lastScore);
            }
        }

        //----- Performance Info -----

        //TODO i18n; reuse?
        String[][] perfInfo = new String[][]{
                {i18N.getMessage("train.overview.perftable.startTime"), ""},
                {i18N.getMessage("train.overview.perftable.totalRuntime"), ""},
                {i18N.getMessage("train.overview.perftable.lastUpdate"), ""},
                {i18N.getMessage("train.overview.perftable.totalParamUpdates"), ""},
                {i18N.getMessage("train.overview.perftable.updatesPerSec"), ""},
                {i18N.getMessage("train.overview.perftable.examplesPerSec"), ""}
        };

        if (last != null) {
            perfInfo[2][1] = String.valueOf(last.getTimeStamp());   //TODO FORMATTING
            perfInfo[3][1] = String.valueOf(last.getTotalMinibatches());
            perfInfo[4][1] = String.valueOf(last.getMinibatchesPerSecond());    //TODO FORMATTING
            perfInfo[5][1] = String.valueOf(last.getExamplesPerSecond());    //TODO FORMATTING
        }

        result.put("perf", perfInfo);


        // ----- Model Info -----
        String[][] modelInfo = new String[][]{
                {i18N.getMessage("train.overview.modeltable.modeltype"), ""},
                {i18N.getMessage("train.overview.modeltable.nLayers"), ""},
                {i18N.getMessage("train.overview.modeltable.nParams"), ""}
        };
        if(!noData) {
            Persistable p = ss.getStaticInfo(currentSessionID, StatsListener.TYPE_ID, wid);
            if (p != null) {
                StatsInitializationReport initReport = (StatsInitializationReport) p;
                int nLayers = initReport.getModelNumLayers();
                long numParams = initReport.getModelNumParams();
                String className = initReport.getModelClassName();

                String modelType;
                if (className.endsWith("MultiLayerNetwork")) {
                    modelType = "MultiLayerNetwork";
                } else if (className.endsWith("ComputationGraph")) {
                    modelType = "ComputationGraph";
                } else {
                    modelType = className;
                }

                modelInfo[0][1] = modelType;
                modelInfo[1][1] = String.valueOf(nLayers);
                modelInfo[2][1] = String.valueOf(numParams);
            }
        }

        result.put("model", modelInfo);

        return Results.ok(Json.toJson(result));
    }
}
