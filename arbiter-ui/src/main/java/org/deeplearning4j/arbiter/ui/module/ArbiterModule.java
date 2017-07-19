package org.deeplearning4j.arbiter.ui.module;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.ui.api.FunctionType;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import play.mvc.Result;
import play.mvc.Results;


import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * Created by Alex on 18/07/2017.
 */
public class ArbiterModule implements UIModule {
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
    public void onAttach(StatsStorage statsStorage) {

    }

    @Override
    public void onDetach(StatsStorage statsStorage) {

    }


    private Result getMainArbiterPage(){


        return Results.ok("Main Arbiter page here");
    }

    private Result getModelResult(String id){

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

        return Results.ok("Result goes here");
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

        return Results.ok("Last update time goes here");
    }

    private Result getModelLastUpdateTimes(String modelIDs){

        return Results.ok("Last update times by model go here");
    }

    private Result getUpdate(String candidateId){

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

        return Results.ok("Candidate results goes here");
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

        return Results.ok("Optimization config goes here");
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

        return Results.ok("Summary results go here");
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

        return Results.ok("Summary results go here");
    }

}
