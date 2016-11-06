package org.deeplearning4j.ui.module.defaultModule;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.ui.api.FunctionType;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;

import java.util.Collection;
import java.util.Collections;
import java.util.List;

import static play.mvc.Results.ok;

/**
 * Created by Alex on 25/10/2016.
 */
public class DefaultModule implements UIModule {
    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.emptyList();
    }

    @Override
    public List<Route> getRoutes() {
        Route r = new Route("/", HttpMethod.GET, FunctionType.Supplier, () -> ok(org.deeplearning4j.ui.views.html.defaultPage.DefaultPage.apply()));

        return Collections.singletonList(r);
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
}
