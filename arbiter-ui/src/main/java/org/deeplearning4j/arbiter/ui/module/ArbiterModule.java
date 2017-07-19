package org.deeplearning4j.arbiter.ui.module;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.arbiter.ui.views.html.ArbiterUI;
import org.deeplearning4j.ui.api.FunctionType;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.stats.StatsListener;
import play.libs.Json;
import play.mvc.Result;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

import static play.mvc.Results.ok;

/**
 * Created by Alex on 18/07/2017.
 */
@Slf4j
public class ArbiterModule implements UIModule {

    public static final String ARBITER_UI_TYPE_ID = "ArbiterUI";

    private Map<String, StatsStorage> knownSessionIDs = Collections.synchronizedMap(new LinkedHashMap<>());
    private String currentSessionID;

    private AtomicLong lastUpdateTime = new AtomicLong(-1);

    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.emptyList();
    }

    @Override
    public List<Route> getRoutes() {
        Route r1 = new Route("/arbiter", HttpMethod.GET, FunctionType.Supplier, this::getMainArbiterPage);
        Route r2 = new Route("/arbiter/modelResults/:id", HttpMethod.GET, FunctionType.Function, this::getModelResult);
        Route r3 = new Route("/arbiter/lastUpdate", HttpMethod.GET, FunctionType.Supplier, this::getLastUpdateTime);
        Route r4 = new Route("/arbiter/lastUpdate/:ids", HttpMethod.GET, FunctionType.Function, this::getModelLastUpdateTimes);
        Route r5 = new Route("/arbiter/update/:id", HttpMethod.GET, FunctionType.Function, this::getUpdate);
        Route r6 = new Route("/arbiter/config", HttpMethod.GET, FunctionType.Supplier, this::getOptimizationConfig);
        Route r7 = new Route("/arbiter/results", HttpMethod.GET, FunctionType.Supplier, this::getSummaryResults);
        Route r8 = new Route("/arbiter/summaryStatus", HttpMethod.GET, FunctionType.Supplier, this::getSummaryStatus);

        return Arrays.asList(r1, r2, r3, r4, r5, r6, r7, r8);
    }

    @Override
    public void reportStorageEvents(Collection<StatsStorageEvent> events) {

    }

    @Override
    public synchronized void onAttach(StatsStorage statsStorage) {
        for (String sessionID : statsStorage.listSessionIDs()) {
            for (String typeID : statsStorage.listTypeIDsForSession(sessionID)) {
                if (!StatsListener.TYPE_ID.equals(typeID))
                    continue;
                knownSessionIDs.put(sessionID, statsStorage);
            }
        }

        if (currentSessionID == null)
            getDefaultSession();
    }

    private void getDefaultSession() {
        if (currentSessionID != null)
            return;

        long mostRecentTime = Long.MIN_VALUE;
        String sessionID = null;
        for (Map.Entry<String, StatsStorage> entry : knownSessionIDs.entrySet()) {
            List<Persistable> staticInfos = entry.getValue().getAllStaticInfos(entry.getKey(), StatsListener.TYPE_ID);
            if (staticInfos == null || staticInfos.size() == 0)
                continue;
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

    @Override
    public void onDetach(StatsStorage statsStorage) {
        for (String s : knownSessionIDs.keySet()) {
            if (knownSessionIDs.get(s) == statsStorage) {
                knownSessionIDs.remove(s);
            }
        }
    }


    private Result getMainArbiterPage(){


//        return Results.ok("Main Arbiter page here");
        return ok(ArbiterUI.apply());
    }

    private Result getModelResult(String id){

        if(currentSessionID == null){
            return ok();
        }

        StatsStorage ss = knownSessionIDs.get(currentSessionID);
        if(ss == null){
            log.warn("Session ID is unknown: {}", currentSessionID);
            return ok();
        }

        return ok("Result for model " + id + " goes here");

        /*
        private Map<Integer,Component> map = new ConcurrentHashMap<>();
        private static final Component NOT_FOUND = new ComponentText("(Candidate results: Not found)",null);
        private final Map<Integer,Long> lastUpdateTimeMap = new ConcurrentHashMap<>();

        @GET
        @Path("/{id}")
        public Response getModelStatus(@PathParam("id") int candidateId){
            if(!map.containsKey(candidateId)) return Response.ok(NOT_FOUND).build();
            String str = "";
            try{
                Component c = map.get(candidateId);
                str = JsonMapper.getMapper().writeValueAsString(c);
            } catch (Exception e){
                if(warnCount.getAndIncrement() < maxWarnCount){
                    log.warn("Error getting candidate info", e);
                }
            }
            return Response.ok(str).build();
        }
         */
    }

    private Result getLastUpdateTime(){

        /*
            private Map<Integer,Component> map = new ConcurrentHashMap<>();
            private static final Component NOT_FOUND = new ComponentText("(Candidate results: Not found)",null);
            private final Map<Integer,Long> lastUpdateTimeMap = new ConcurrentHashMap<>();

            @GET
            @Path("/lastUpdate")
            @Consumes(MediaType.APPLICATION_JSON)
            @Produces(MediaType.APPLICATION_JSON)
            public Response getLastUpdateTimes(@QueryParam("id")List<String> modelIDs){
                //Here: requests are of the form /modelResults/lastUpdate?id=0&id=1&id=2
                Map<Integer,Long> outMap = new HashMap<>();
                for( String id : modelIDs ){
                    if(!lastUpdateTimeMap.containsKey(id)) outMap.put(Integer.valueOf(id),0L);
                    else outMap.put(Integer.valueOf(id),lastUpdateTimeMap.get(id));
                }
                return Response.ok(outMap).build();
            }
         */

        return ok(String.valueOf(lastUpdateTime.get()));
    }

    private Result getModelLastUpdateTimes(String modelIDs){

        if(currentSessionID == null){
            return ok();
        }

        StatsStorage ss = knownSessionIDs.get(currentSessionID);
        if(ss == null){
            log.warn("getModelLastUpdateTimes(): Session ID is unknown: {}", currentSessionID);
            return ok("-1");
        }

        String[] split = modelIDs.split(",");

        long[] lastUpdateTimes = new long[split.length];
        for( int i=0; i<split.length; i++ ){
            String s = split[i];
            Persistable p = ss.getLatestUpdate(currentSessionID, ARBITER_UI_TYPE_ID, s);
            if(p != null){
                lastUpdateTimes[i] = p.getTimeStamp();
            }
        }

        return ok(Json.toJson(lastUpdateTimes));
    }

    private Result getUpdate(String candidateId){

        StatsStorage ss = knownSessionIDs.get(currentSessionID);
        if(ss == null){
            log.warn("getModelLastUpdateTimes(): Session ID is unknown: {}", currentSessionID);
            return ok();
        }



        /*
            private Map<Integer,Component> map = new ConcurrentHashMap<>();
            private static final Component NOT_FOUND = new ComponentText("(Candidate results: Not found)",null);
            private final Map<Integer,Long> lastUpdateTimeMap = new ConcurrentHashMap<>();

            @POST
            @Path("/update/{id}")
            @Consumes(MediaType.TEXT_PLAIN)
            @Produces(MediaType.APPLICATION_JSON)
            public Response update(@PathParam("id")int candidateID, String componentStr){

                if(componentStr == null || componentStr.isEmpty()){
                    return Response.ok(Collections.singletonMap("status", "ok")).build();
                }

                try{
                    Component component = JsonMapper.getMapper().readValue(componentStr, Component.class);
                    map.put(candidateID,component);
                } catch (Exception e) {
                    if(warnCount.getAndIncrement() < maxWarnCount){
                        log.warn("Error posting summary status update", e);
                    }
                }

                return Response.ok(Collections.singletonMap("status", "ok")).build();
            }
         */

        return ok("Candidate results goes here");
    }

    private Result getOptimizationConfig(){
        /*
        private Component component = null;

        private static final int maxWarnCount = 5;
        private AtomicInteger warnCount = new AtomicInteger(0);

        @GET
        public Response getConfig(){
            log.trace("GET for config with current component: {}",component);

            String str = "";
            try{
                str = JsonMapper.getMapper().writeValueAsString(component);
            } catch (Exception e){
                if(warnCount.getAndIncrement() < maxWarnCount){
                    log.warn("Error getting configuration UI info", e);
                }
            }
            Response r = Response.ok(str).build();
            return r;
        }
         */

        return ok("Optimization config goes here");
    }

    private Result getSummaryResults(){
        /*
            private List<CandidateStatus> statusList = new ArrayList<>();

            @GET
            public Response getCandidateStatus(){
                log.trace("GET for candidate status with current status: {}",statusList);

                return Response.ok(statusList).build();
            }
         */

        return ok("Summary results go here");
    }

    private Result getSummaryStatus(){
        /*
            private Component component = null;

            @GET
            public Response getStatus(){
                log.trace("Get with elements: {}",component);
                String str = "";
                try{
                    str = JsonMapper.getMapper().writeValueAsString(component);
                } catch (Exception e){
                    if(warnCount.getAndIncrement() < maxWarnCount){
                        log.warn("Error getting summary status update", e);
                    }
                }
                Response r = Response.ok(str).build();
                return r;
            }
         */

        return ok("Summary results go here");
    }

}
