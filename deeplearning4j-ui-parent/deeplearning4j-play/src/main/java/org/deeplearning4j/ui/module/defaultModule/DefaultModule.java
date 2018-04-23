package org.deeplearning4j.ui.module.defaultModule;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.ui.api.FunctionType;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.i18n.I18NResource;

import java.io.File;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import static play.mvc.Results.redirect;

/**
 * Landing page - i.e., "/" route
 * @author Alex Black
 */
public class DefaultModule implements UIModule {
    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.emptyList();
    }

    @Override
    public List<Route> getRoutes() {
        //TODO
        //        Route r = new Route("/", HttpMethod.GET, FunctionType.Supplier, () -> ok(org.deeplearning4j.ui.views.html.defaultPage.DefaultPage.apply()));
        Route r = new Route("/", HttpMethod.GET, FunctionType.Supplier, () -> redirect("/train/overview"));

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

    @Override
    public List<I18NResource> getInternationalizationResources() {
        return Collections.emptyList();
    }
}
