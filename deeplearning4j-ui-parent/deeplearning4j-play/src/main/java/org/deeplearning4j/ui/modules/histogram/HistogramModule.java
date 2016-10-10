package org.deeplearning4j.ui.modules.histogram;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.ui.api.FunctionType;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.stats.api.StatsInitializationReport;
import org.deeplearning4j.ui.stats.api.StatsReport;
import org.deeplearning4j.ui.stats.api.StatsType;
import org.deeplearning4j.ui.stats.api.SummaryType;
import org.deeplearning4j.ui.storage.Persistable;
import org.deeplearning4j.ui.storage.StatsStorage;
import org.deeplearning4j.ui.storage.StatsStorageEvent;

import org.deeplearning4j.ui.views.html.histogram.Histogram;
import org.deeplearning4j.ui.weights.beans.CompactModelAndGradient;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import play.api.http.ContentTypes;
import play.libs.Json;
import play.mvc.Result;
import play.mvc.Results;

import java.util.*;

import static play.mvc.Results.ok;

/**
 * Created by Alex on 08/10/2016.
 */
@Slf4j
public class HistogramModule implements UIModule {

    private ObjectMapper om = new ObjectMapper();
    private boolean initialized = false;

    private Map<String,StatsStorage> knownSessionIDs = new LinkedHashMap<>();

//    private ArrayList<Double> scoreHistory = new ArrayList<>();
//    private List<Map<String,List<Double>>> meanMagHistoryParams = new ArrayList<>();    //1 map per layer; keyed by new param name
//    private List<Map<String,List<Double>>> meanMagHistoryUpdates = new ArrayList<>();
//    private Map<String,Integer> layerNameIndexes = new HashMap<>();
//    private List<String> layerNames = new ArrayList<>();
//    private int layerNameIndexesCount = 0;


    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.singletonList(StatsListener.TYPE_ID);
    }

    @Override
    public List<Route> getRoutes() {
        Route r = new Route("/weights", HttpMethod.GET, FunctionType.Supplier, () -> ok(org.deeplearning4j.ui.views.html.histogram.Histogram.apply()));
        Route r2 = new Route("/weights/listSessions", HttpMethod.GET, FunctionType.Supplier, () -> ok(Json.toJson(knownSessionIDs.keySet())));
        Route r3 = new Route("/weights/updated/:sid", HttpMethod.GET, FunctionType.Function, this::getLastUpdateTime);
        Route r4 = new Route("/weights/data/:sid", HttpMethod.GET, FunctionType.Function, this::processRequest);

        return Arrays.asList(r, r2, r3, r4);
    }

    @Override
    public void reportStorageEvents(StatsStorage statsStorage, Collection<StatsStorageEvent> events) {
        log.info("Received events: {}",events);

        //We should only be getting relevant session IDs...
        for(StatsStorageEvent sse : events){
            if(!knownSessionIDs.containsKey(sse.getSessionID())){
                knownSessionIDs.put(sse.getSessionID(), statsStorage);
            }
        }

        //TODO only do updates on demand... no point updating things over and over if nobody is actually looking at the page!
    }

    @Override
    public void onAttach(StatsStorage statsStorage) {
        for(String sessionID : statsStorage.listSessionIDs()){
            for(String typeID : statsStorage.listTypeIDsForSession(sessionID)){
                if(!StatsListener.TYPE_ID.equals(typeID)) continue;
                knownSessionIDs.put(sessionID, statsStorage);
            }
        }
    }

    @Override
    public void onDetach(StatsStorage statsStorage) {
        for(String sessionID : statsStorage.listSessionIDs()){
            knownSessionIDs.remove(sessionID);
        }
    }

    private Result getLastUpdateTime(String sessionID){
        return Results.ok(Json.toJson(System.currentTimeMillis()));
    }

    private Result processRequest(String sessionId){
        StatsStorage ss = knownSessionIDs.get(sessionId);
        if(ss == null){
            return Results.notFound("Unknown session ID: " + sessionId);
        }

        List<String> workerIDs = ss.listWorkerIDsForSession(sessionId);

        //TODO checks
        StatsInitializationReport initReport = (StatsInitializationReport)ss.getStaticInfo(sessionId,StatsListener.TYPE_ID, workerIDs.get(0));
        if(initReport == null) return Results.ok(Json.toJson(Collections.EMPTY_MAP));

//        String[] paramNames = initReport.getModelParamNames();

        List<Persistable> list = ss.getAllUpdatesAfter(sessionId, StatsListener.TYPE_ID, workerIDs.get(0),0);
        Collections.sort(list, (a,b) -> Long.compare(a.getTimeStamp(), b.getTimeStamp()));

        List<Double> scoreList = new ArrayList<>(list.size());
        List<Map<String,List<Double>>> meanMagHistoryParams = new ArrayList<>();    //List.get(i) -> layer i
        StatsReport last = null;
        for(Persistable p : list){
            if(!(p instanceof StatsReport)){
                log.warn("Unexpected type: {}", p);
                continue;
            }
            StatsReport sp = (StatsReport)p;
            scoreList.add(sp.getScore());

            //TODO mean magnitudes

            last = sp;
        }

        Map<String,Map> newParams = getHistogram(last.getHistograms(StatsType.Parameters));
        Map<String,Map> newGrad = getHistogram(last.getHistograms(StatsType.Updates));

        double lastScore = (scoreList.size() == 0 ? 0.0 : scoreList.get(scoreList.size()-1));

        CompactModelAndGradient g = new CompactModelAndGradient();
        g.setGradients(newGrad);
        g.setParameters(newParams);
        g.setScore(lastScore);
        g.setScores(scoreList);
//        g.setPath(subPath);
//        g.setUpdateMagnitudes(meanMagHistoryUpdates);
//        g.setParamMagnitudes(meanMagHistoryParams);
//        g.setLayerNames(layerNames);
        g.setLastUpdateTime(last.getTimeStamp());

//        Map<String,Object> ret = new HashMap<>();
//        ret.put("lastUpdateTime", last.getTimeStamp());
//        ret.put("score",lastScore);
//        ret.put("scores",scoreList);
//        ret.put("gradients", newGrad);
//        ret.put("parameters", newParams);
//        ret.put("paramMagnitudes", Collections.EMPTY_MAP);
//        ret.put("updateMagnitudes", Collections.EMPTY_MAP);


//        return Results.ok(Json.toJson(ret));
        return Results.ok(Json.toJson(g));
    }

    private Map<String,Map> getHistogram(Map<String, org.deeplearning4j.ui.stats.api.Histogram> histograms){
        Map<String,Map> ret = new LinkedHashMap<>();
        for(String s : histograms.keySet()){
            org.deeplearning4j.ui.stats.api.Histogram h = histograms.get(s);
            String newName;
            if(Character.isDigit(s.charAt(0))) newName = "param_" + s;
            else newName = s;

            Map<Number,Number> temp = new LinkedHashMap<>();
            double min = h.getMin();
            double max = h.getMax();
            int n = h.getNBins();
            double step = (max-min)/n;
            int[] counts = h.getBinCounts();
            for( int i=0; i<n; i++ ){
                double binLoc = min + i*step + step/2.0;
                temp.put(binLoc, counts[i]);
            }

            ret.put(newName,temp);
        }
        return ret;
    }
}
