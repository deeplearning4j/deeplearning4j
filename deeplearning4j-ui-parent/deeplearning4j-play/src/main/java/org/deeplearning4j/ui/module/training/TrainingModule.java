package org.deeplearning4j.ui.module.training;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.ui.api.FunctionType;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.i18n.I18NProvider;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.views.html.training.Training;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import static play.mvc.Results.ok;

/**
 * Created by Alex on 14/10/2016.
 */
public class TrainingModule implements UIModule {
    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.singletonList(StatsListener.TYPE_ID);
    }

    @Override
    public List<Route> getRoutes() {
        Route r = new Route("/train", HttpMethod.GET, FunctionType.Supplier, () -> ok(Training.apply(I18NProvider.getInstance())));

        return Arrays.asList(r);
    }

    @Override
    public void reportStorageEvents(StatsStorage statsStorage, Collection<StatsStorageEvent> events) {

    }

    @Override
    public void onAttach(StatsStorage statsStorage) {

    }

    @Override
    public void onDetach(StatsStorage statsStorage) {

    }
}
