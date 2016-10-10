package org.deeplearning4j.ui.modules.histogram;

import org.deeplearning4j.ui.api.FunctionType;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.StatsStorage;
import org.deeplearning4j.ui.storage.StatsStorageEvent;

import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.util.*;

import static play.mvc.Results.ok;

/**
 * Created by Alex on 08/10/2016.
 */
public class HistogramModule implements UIModule {

    private ObjectMapper om = new ObjectMapper();
    private boolean initialized = false;

    private Set<String> knownSessionIDs = new LinkedHashSet<>();

    private ArrayList<Double> scoreHistory = new ArrayList<>();
    private List<Map<String,List<Double>>> meanMagHistoryParams = new ArrayList<>();    //1 map per layer; keyed by new param name
    private List<Map<String,List<Double>>> meanMagHistoryUpdates = new ArrayList<>();
    private Map<String,Integer> layerNameIndexes = new HashMap<>();
    private List<String> layerNames = new ArrayList<>();
    private int layerNameIndexesCount = 0;


    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.singletonList(StatsListener.TYPE_ID);
    }

    @Override
    public List<Route> getRoutes() {
        Route r = new Route("/weights", HttpMethod.GET, FunctionType.Supplier, () -> ok(org.deeplearning4j.ui.views.html.histogram.Histogram.apply()));
        Route r2 = new Route("/weights/listSessions", HttpMethod.GET, FunctionType.Supplier, () -> ok(toJson(knownSessionIDs)));

        return Arrays.asList(r, r2);
    }

    @Override
    public void reportStorageEvents(StatsStorage statsStorage, Collection<StatsStorageEvent> events) {
        //TODO actually process the events...
        if(!initialized) doInit(statsStorage);

        //TODO only do updates on demand... no point updating things over and over if nobody is actually looking at the page!


    }

    @Override
    public void onAttach(StatsStorage statsStorage) {
        for(String sessionID : statsStorage.listSessionIDs()){
            for(String typeID : statsStorage.listTypeIDsForSession(sessionID)){
                if(!StatsListener.TYPE_ID.equals(typeID)) continue;
                knownSessionIDs.add(sessionID);
            }
        }

        //Query the stats storage
        if(!initialized) doInit(statsStorage);
    }

    @Override
    public void onDetach(StatsStorage statsStorage) {
        //TODO
        for(String sessionID : statsStorage.listSessionIDs()){
            knownSessionIDs.remove(sessionID);
        }
    }

    private void doInit(StatsStorage statsStorage){
        //Query the stats storage... determine if any relevant stats are available
        //If so, record the session/worker IDs for later use
        //Old histogram listener (that this is based on)

        List<String> sessionIDs = statsStorage.listSessionIDs();
        for(String sid : sessionIDs){
//            statsStorage.get
        }


    }

    private String toJson(Object object){
        try{
            return om.writeValueAsString(object);
        }catch (Exception e){
            return "Json Processing Exception"; //TODO
        }
    }
}
