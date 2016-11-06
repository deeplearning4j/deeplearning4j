package org.deeplearning4j.ui.module.tsne;

import com.fasterxml.jackson.databind.JsonNode;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.ui.api.FunctionType;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import play.mvc.Http;
import play.mvc.Result;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import static play.mvc.Controller.request;
import static play.mvc.Results.ok;

/**
 * Created by Alex on 25/10/2016.
 */
public class TsneModule implements UIModule {


    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.emptyList();
    }

    @Override
    public List<Route> getRoutes() {
        Route r1 = new Route("/tsne", HttpMethod.GET, FunctionType.Supplier, () -> ok(org.deeplearning4j.ui.views.html.tsne.Tsne.apply()));
        Route r2 = new Route("/tsne/posttest", HttpMethod.POST, FunctionType.Supplier, this::testPost );
        return Arrays.asList(r1, r2);
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

    private Result testPost(){
        Http.RequestBody body = request().body();

        if(body != null) {
            JsonNode json = body.asJson();

            System.out.println("GOT: " + json);
            System.out.println("GOT2: " + body);
        } else {
            System.out.println("GOT: " + null);
        }

        return ok("Got text: " + body.asText());
    }
}
