package org.deeplearning4j.ui.module.tsne;

import com.fasterxml.jackson.databind.JsonNode;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.ui.api.FunctionType;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import play.libs.Json;
import play.mvc.Http;
import play.mvc.Result;
import play.mvc.Results;

import java.util.*;

import static play.mvc.Controller.request;
import static play.mvc.Results.ok;

/**
 * Created by Alex on 25/10/2016.
 */
public class TsneModule implements UIModule {

    private Map<String, StatsStorage> knownSessionIDs = Collections.synchronizedMap(new LinkedHashMap<>());

    public TsneModule(){

    }

    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.emptyList();
    }

    @Override
    public List<Route> getRoutes() {
        Route r1 = new Route("/tsne", HttpMethod.GET, FunctionType.Supplier, () -> ok(org.deeplearning4j.ui.views.html.tsne.Tsne.apply()));
        Route r2 = new Route("/tsne/sessions", HttpMethod.GET, FunctionType.Supplier, this::listSessions);
        Route r3 = new Route("/tsne/coords/:sid", HttpMethod.GET, FunctionType.Function, this::getCoords);
//        Route r2 = new Route("/tsne/posttest", HttpMethod.POST, FunctionType.Supplier, this::testPost );
        return Arrays.asList(r1, r2, r3);
//        return Collections.emptyList();
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

    private Result listSessions(){
        return Results.ok(Json.toJson(knownSessionIDs.keySet()));
    }

    private Result getCoords(String sessionId){
        String[] lines = new String[]{
                "-3775.297235325235,24673.113908278356,play",
                "-8812.320495393104,8763.036758305949,dur",
                "23811.361013839844,-18446.265486393168,hi",
                "-22399.729196208387,21034.4598526551,big",
                "-13728.129868038637,-22419.653625552575,year",
                "3570.5427808768964,-31919.751981267204,ago",
                "-11663.727519330943,-6622.964118228104,american",
                "19318.2404644759,-26774.8668631964,program",
                "-21396.754952842883,3703.2177212660786,three",
                "-34298.79759281683,-9244.626743418836,music",
                "32808.359103628034,-11482.704968054868,children",
                "-14987.261240502326,-4175.681475451747,four",
                "-38251.54854224372,-6530.951461377245,season",
                "-9961.289770846204,18323.141788279132,state",
                "18142.687768792675,4495.584390685736,case",
                "-35019.31396532761,4173.793773420839,still",
                "5765.519916483223,2105.9875585256273,$",
                "29050.73682425,4063.113025980962,STOP",
                "38225.39635789529,4889.97045555356,made",
                "-37854.67814698981,154.92299417458244,work",
                "-8581.382494234285,2545.611018731874,old",
                "20536.236672898096,-22840.8935490519,director",
                "-8711.389373751175,-28883.119880550028,'",
                "14364.88744695551,1937.8692653260607,want",
                "-12205.73030481528,36323.00365351727,night"
        };

        return Results.ok(Json.toJson(lines));
    }
}
