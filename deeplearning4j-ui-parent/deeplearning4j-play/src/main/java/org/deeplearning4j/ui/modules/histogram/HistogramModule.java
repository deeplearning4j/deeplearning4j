package org.deeplearning4j.ui.modules.histogram;

import org.deeplearning4j.ui.api.FunctionType;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.StatsStorage;
import org.deeplearning4j.ui.storage.StatsStorageEvent;

import java.util.Collection;
import java.util.Collections;
import java.util.List;

import static play.mvc.Results.ok;

/**
 * Created by Alex on 08/10/2016.
 */
public class HistogramModule implements UIModule {
    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.singletonList(StatsListener.TYPE_ID);
    }

    @Override
    public List<Route> getRoutes() {
        Route r = new Route("/histogram", HttpMethod.GET, FunctionType.Supplier, () -> ok("This is the histogram page"));
        return Collections.singletonList(r);
    }

    @Override
    public void reportStorageEvents(StatsStorage statsStorage, Collection<StatsStorageEvent> events) {
        //TODO actually process the events...


        
    }


}
