package org.deeplearning4j.arbiter.ui.module;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.api.storage.StatsStorageListener;
import org.deeplearning4j.arbiter.BaseNetworkSpace;
import org.deeplearning4j.arbiter.layers.LayerSpace;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.runner.CandidateStatus;
import org.deeplearning4j.arbiter.ui.UpdateStatus;
import org.deeplearning4j.arbiter.ui.data.GlobalConfigPersistable;
import org.deeplearning4j.arbiter.ui.data.ModelInfoPersistable;
import org.deeplearning4j.arbiter.ui.misc.UIUtils;
import org.deeplearning4j.arbiter.ui.views.html.ArbiterUI;
import org.deeplearning4j.ui.api.*;
import org.deeplearning4j.ui.api.Component;
import org.deeplearning4j.ui.components.chart.ChartLine;
import org.deeplearning4j.ui.components.chart.ChartScatter;
import org.deeplearning4j.ui.components.chart.style.StyleChart;
import org.deeplearning4j.ui.components.component.ComponentDiv;
import org.deeplearning4j.ui.components.component.style.StyleDiv;
import org.deeplearning4j.ui.components.table.ComponentTable;
import org.deeplearning4j.ui.components.table.style.StyleTable;
import org.deeplearning4j.ui.components.text.ComponentText;
import org.deeplearning4j.ui.components.text.style.StyleText;
import org.joda.time.DurationFieldType;
import org.joda.time.Period;
import org.joda.time.PeriodType;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;
import play.libs.Json;
import play.mvc.Result;

import java.awt.*;
import java.util.*;
import java.util.List;

import static org.deeplearning4j.arbiter.ui.misc.JsonMapper.asJson;
import static play.mvc.Results.ok;

/**
 * Created by Alex on 18/07/2017.
 */
@Slf4j
public class ArbiterModule implements UIModule {

    private static final DateTimeFormatter TIME_FORMATTER = DateTimeFormat.forPattern("YYYY-MM-dd HH:mm ZZ");
    public static final String ARBITER_UI_TYPE_ID = "ArbiterUI";

    private static final String JSON = "application/json";

    private Map<String, StatsStorage> knownSessionIDs = Collections.synchronizedMap(new LinkedHashMap<>());
    private String currentSessionID;

    private Map<String, Long> lastUpdateForSession = Collections.synchronizedMap(new HashMap<>());


    //Styles for UI:
    private static final StyleTable STYLE_TABLE = new StyleTable.Builder()
            .width(100, LengthUnit.Percent)
            .backgroundColor(Color.WHITE)
            .borderWidth(1)
            .columnWidths(LengthUnit.Percent, 30, 70)
            .build();

    private static final StyleDiv STYLE_DIV_WIDTH_100_PC = new StyleDiv.Builder()
            .width(100, LengthUnit.Percent)
            .build();

    private static final ComponentDiv DIV_SPACER_20PX = new ComponentDiv(new StyleDiv.Builder()
            .width(100,LengthUnit.Percent)
            .height(20, LengthUnit.Px).build());

    private static final StyleChart STYLE_CHART = new StyleChart.Builder()
            .width(650, LengthUnit.Px)
            .height(350, LengthUnit.Px)
            .build();


    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.singletonList(ARBITER_UI_TYPE_ID);
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
        Route r8 = new Route("/arbiter/summary", HttpMethod.GET, FunctionType.Supplier, this::getSummaryStatus);

        return Arrays.asList(r1, r2, r3, r4, r5, r6, r7, r8);
    }

    @Override
    public void reportStorageEvents(Collection<StatsStorageEvent> events) {
        log.info("Reported storage events: {}", events);

        for (StatsStorageEvent sse : events) {
            if (ARBITER_UI_TYPE_ID.equals(sse.getTypeID())) {
                if (sse.getEventType() == StatsStorageListener.EventType.PostStaticInfo) {
                    knownSessionIDs.put(sse.getSessionID(), sse.getStatsStorage());
                }

                Long lastUpdate = lastUpdateForSession.get(sse.getSessionID());
                if (lastUpdate == null) {
                    lastUpdateForSession.put(sse.getSessionID(), sse.getTimestamp());
                } else if (sse.getTimestamp() > lastUpdate) {
                    lastUpdateForSession.put(sse.getSessionID(), sse.getTimestamp()); //Should be thread safe - read only elsewhere
                }
            }
        }

        if (currentSessionID == null)
            getDefaultSession();

        if(currentSessionID == null){
            getDefaultSession();
        }
    }

    @Override
    public synchronized void onAttach(StatsStorage statsStorage) {
        log.info("Attached: {}", statsStorage);
        for (String sessionID : statsStorage.listSessionIDs()) {
            for (String typeID : statsStorage.listTypeIDsForSession(sessionID)) {
                if (!ARBITER_UI_TYPE_ID.equals(typeID))
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
            List<Persistable> staticInfos = entry.getValue().getAllStaticInfos(entry.getKey(), ARBITER_UI_TYPE_ID);
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

        log.info("Default session is: {}", currentSessionID);
    }

    @Override
    public void onDetach(StatsStorage statsStorage) {
        log.info("Detached: {}", statsStorage);
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
            log.warn("getMainArbiterPage(): Session ID is unknown: {}", currentSessionID);
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

        //TODO
        long t = System.currentTimeMillis();
        UpdateStatus us = new UpdateStatus(t, t, t);

        return ok(Json.toJson(us));
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


        StatsStorage ss = knownSessionIDs.get(currentSessionID);
        if(ss == null){
            log.warn("getOptimizationConfig(): Session ID is unknown: {}", currentSessionID);
            return ok();
        }

        Persistable p = ss.getStaticInfo(currentSessionID, ARBITER_UI_TYPE_ID, GlobalConfigPersistable.GLOBAL_WORKER_ID);

        if(p == null){
            log.info("No static info");
            return ok();
        }

        List<Component> components = new ArrayList<>();

        GlobalConfigPersistable gcp = (GlobalConfigPersistable)p;
        OptimizationConfiguration oc = gcp.getOptimizationConfiguration();

        //Here: report optimization settings/configuration.


        String[] tableHeader = {"Configuration", "Value"};

        String[][] table = new String[][]{
                {"Candidate Generator", oc.getCandidateGenerator().getClass().getSimpleName()},
                {"Data Provider", oc.getDataProvider().toString()},
                {"Score Function", oc.getScoreFunction().toString()},
                {"Result Saver", oc.getResultSaver().toString()},
//                {"Model Hyperparameter Space", oc.getCandidateGenerator().getParameterSpace().toString()}

        };

        ComponentTable ct = new ComponentTable.Builder(STYLE_TABLE)
                .content(table)
                .header(tableHeader)
                .build();
        components.add(ct);


        BaseNetworkSpace<?> ps = (BaseNetworkSpace)oc.getCandidateGenerator().getParameterSpace();
        Map<String,ParameterSpace<?>> m = ps.getGlobalConfigAsMap();

        String[][] hSpaceTable = new String[m.size()][2];
        int i=0;
        for(Map.Entry<String,ParameterSpace<?>> e : m.entrySet()){
            hSpaceTable[i][0] = e.getKey();
            hSpaceTable[i][1] = e.getValue().toString();
            i++;
        }

        components.add(DIV_SPACER_20PX);

        String[] hSpaceTableHeader = new String[]{"Hyperparameter", "Hyperparameter Configuration"};

        ComponentTable ct2 = new ComponentTable.Builder(STYLE_TABLE)
                .content(hSpaceTable)
                .header(hSpaceTableHeader)
                .build();



        components.add(ct2);

        List<BaseNetworkSpace.LayerConf> layerConfs = ps.getLayerSpaces();


        StyleText sText = new StyleText.Builder()
                .fontSize(12)
                .build();
//        components.add(new ComponentText.Builder(" -- Layer Spaces --",sText).build());

        for(BaseNetworkSpace.LayerConf l : layerConfs){
            LayerSpace<?> ls = l.getLayerSpace();
            Map<String,ParameterSpace<?>> lpsm = ls.getConfigAsMap();

            String[][] t = new String[lpsm.size()][2];
            i=0;
            for(Map.Entry<String,ParameterSpace<?>> e : lpsm.entrySet()){
                t[i][0] = e.getKey();
                t[i][1] = e.getValue().toString();
                i++;
            }

            ComponentTable ct3 = new ComponentTable.Builder(STYLE_TABLE)
                    .content(t)
                    .header(hSpaceTableHeader)
                    .build();

            String title = "Layer Space: " + ls.getClass().getSimpleName() + ", Name: " + l.getLayerName();

            components.add(DIV_SPACER_20PX);
            components.add(new ComponentText.Builder(title, sText).build());
            components.add(ct3);


        }


        ComponentDiv cd = new ComponentDiv(STYLE_DIV_WIDTH_100_PC, components);

        return ok(asJson(cd)).as(JSON);
    }

    private Result getSummaryResults(){
        /*
            private List<CandidateInfo> statusList = new ArrayList<>();

            @GET
            public Response getCandidateStatus(){
                log.trace("GET for candidate status with current status: {}",statusList);

                return Response.ok(statusList).build();
            }
         */

        String[][] temp = new String[][]{
                {"0", "1.0", "Status 0"},
                {"1", "2.0", "Status 1"}

        };

        return ok(asJson(temp)).as(JSON);
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

        //First: table - number completed, queued, running, failed, total
        //Best model index, score, and time
        //Total runtime
        //Termination conditions

        StatsStorage ss = knownSessionIDs.get(currentSessionID);
        if(ss == null){
            log.warn("getOptimizationConfig(): Session ID is unknown: {}", currentSessionID);
            return ok();
        }

        Persistable p = ss.getStaticInfo(currentSessionID, ARBITER_UI_TYPE_ID, GlobalConfigPersistable.GLOBAL_WORKER_ID);

        if(p == null){
            log.info("No static info");
            return ok();
        }

        List<Component> components = new ArrayList<>();

        GlobalConfigPersistable gcp = (GlobalConfigPersistable)p;
        OptimizationConfiguration oc = gcp.getOptimizationConfiguration();
        long execStartTime = oc.getExecutionStartTime();

        List<TerminationCondition> tcs = oc.getTerminationConditions();

        //TODO: I18N

        //TODO don't use currentTimeMillis due to stored data
        String runtime = UIUtils.formatDuration(System.currentTimeMillis() - execStartTime);
        long bestTime = System.currentTimeMillis(); //TODO
        String sinceBest = UIUtils.formatDuration(System.currentTimeMillis() - bestTime);


        String[][] table = new String[][]{
                {"Models Completed", String.valueOf(gcp.getCandidatesCompleted())},
                {"Models Queued/Running", String.valueOf(gcp.getCandidatesQueued())},
                {"Models Failed", String.valueOf(gcp.getCandidatesFailed())},
                {"Models Total", String.valueOf(gcp.getCandidatesTotal())},
                {"Best Score", ""},
                {"Best Scoring Model", "Index x, Found at y (" + sinceBest + " ago)"},
                {"Execution Start Time", TIME_FORMATTER.print(execStartTime)},
                {"Total Runtime", runtime}
        };



        ComponentTable ct = new ComponentTable.Builder(STYLE_TABLE)
                .content(table)
                .header("Status", "")
                .build();

        components.add(ct);

        String[][] tcTable = new String[tcs.size()][2];
        for( int i=0; i<tcs.size(); i++ ){
            tcTable[i][0] = "Termination Condition " + i;
            tcTable[i][1] = tcs.get(i).toString();
        }

        components.add(DIV_SPACER_20PX);

        ComponentTable ct2 = new ComponentTable.Builder(STYLE_TABLE)
                .content(tcTable)
                .header("Termination Condition", "")
                .build();

        components.add(ct2);


        //Charts:
        //Best model score vs. time
        //All candidate scores (scatter plot vs. time)

        //How to get this? query all model infos...

        List<Persistable> allModelInfoTemp = new ArrayList<>(ss.getLatestUpdateAllWorkers(currentSessionID, ARBITER_UI_TYPE_ID));
        List<ModelInfoPersistable> allModelInfo = new ArrayList<>();
        for(Persistable per : allModelInfoTemp){
            ModelInfoPersistable mip = (ModelInfoPersistable)per;
            if(mip.getStatus() == CandidateStatus.Complete && mip.getScore() != null && Double.isFinite(mip.getScore())){
                allModelInfo.add(mip);
            }
        }

        allModelInfo.sort(Comparator.comparingLong(Persistable::getTimeStamp));

        List<Double> bestX = new ArrayList<>();
        List<Double> bestY = new ArrayList<>();

        double[] allX = new double[allModelInfo.size()];
        double[] allY = new double[allModelInfo.size()];

        boolean minimize = oc.getScoreFunction().minimize();

        double bestScore = (minimize ? Double.MAX_VALUE : -Double.MAX_VALUE);
        double worstScore = (minimize ? -Double.MAX_VALUE : Double.MAX_VALUE);
        int bestScoreIdx = -1;
        double bestScoreTime = -1L;
        double lastTime = -1L;
        for(int i=0; i<allModelInfo.size(); i++ ){
            ModelInfoPersistable mip = allModelInfo.get(i);
            double currScore = mip.getScore();
            double t = (execStartTime - mip.getTimeStamp()) / 60000.0;    //60000 ms per minute

            allX[i] = t;
            allY[i] = currScore;

            if(i == 0){
                bestX.add(t);
                bestY.add(currScore);
                bestScore = currScore;
                bestScoreIdx = mip.getModelIdx();
                bestScoreTime = t;
            } else if((!minimize && currScore > bestScore) || (minimize && currScore < bestScore)){
                bestX.add(t);
                bestY.add(bestScore);
                bestX.add(t);  //TODO non-real time rendering support...
                bestY.add(currScore);

                bestScore = currScore;
                bestScoreIdx = mip.getModelIdx();
                bestScoreTime = t;
            }

            if((!minimize && currScore < worstScore) || (minimize && currScore > worstScore)){
                worstScore = currScore;
            }

            if(t > lastTime){
                lastTime = t;
            }
        }

        double[] scatterGraphMinMax = UIUtils.niceRange(Math.max(bestScore, worstScore), Math.min(bestScore, worstScore), 5);

        double[] lineGraphMinMax = UIUtils.niceRange(
                bestY.stream().mapToDouble(s -> s).max().orElse(0),bestY.stream().mapToDouble(s -> s).min().orElse(0), 5
        );

        System.out.println("Orig worst/best: " + worstScore + ", " + bestScore);
        System.out.println("Nice graph range: " + Arrays.toString(scatterGraphMinMax));

        if(bestX.size() > 0) {
            bestX.add((double) lastTime);
            bestY.add(bestY.get(bestY.size() - 1));
        }


        double[] bestXd = new double[bestX.size()];
        double[] bestYd = new double[bestXd.length];
        for( int i=0; i<bestX.size(); i++ ){
            bestXd[i] = bestX.get(i);
            bestYd[i] = bestY.get(i);
        }

        ChartLine cl = new ChartLine.Builder("Best Model Score vs. Time", STYLE_CHART)
                .addSeries("Best Score vs. Time", bestXd, bestYd)
                .setYMin(lineGraphMinMax[0])
                .setYMax(lineGraphMinMax[1])
                .build();
        components.add(cl);

        ChartScatter cs = new ChartScatter.Builder("All Candidate Scores", STYLE_CHART)
                .addSeries("Candidates", allX, allY)
                .setYMin(scatterGraphMinMax[0])
                .setYMax(scatterGraphMinMax[1])
                .build();

        components.add(cs);

        ComponentDiv cd = new ComponentDiv(STYLE_DIV_WIDTH_100_PC, components);




//        System.out.println("Best score chart:");
//        System.out.println(Arrays.toString(bestXd));
//        System.out.println(Arrays.toString(bestYd));
//
//        System.out.println("All score chart:");
//        System.out.println(Arrays.toString(allX));
//        System.out.println(Arrays.toString(allY));

        return ok(asJson(cd)).as(JSON);
    }

}
