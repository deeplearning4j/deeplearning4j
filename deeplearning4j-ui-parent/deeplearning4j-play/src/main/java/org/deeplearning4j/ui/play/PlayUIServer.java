package org.deeplearning4j.ui.play;

import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.modules.histogram.HistogramModule;
import org.deeplearning4j.ui.storage.StatsStorage;
import play.Mode;
import play.mvc.Result;
import play.routing.Router;
import play.routing.RoutingDsl;
import play.server.Server;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

import static play.mvc.Results.ok;

/**
 * Created by Alex on 08/10/2016.
 */
public class PlayUIServer extends UIServer {



    public PlayUIServer(){
        RoutingDsl routingDsl = new RoutingDsl();

        routingDsl.GET("/").routeTo(new Index());

        List<UIModule> uiModules = new ArrayList<>();
        uiModules.add(new HistogramModule());

        for(UIModule m : uiModules){
            List<Route> routes = m.getRoutes();
            for( Route r : routes){
                RoutingDsl.PathPatternMatcher ppm = routingDsl.match(r.getHttpMethod().name(),r.getRoute());
                switch (r.getFunctionType()){
                    case Supplier:
                        ppm.routeTo(r.getSupplier());
                        break;
                    case Function:
                    case BiFunction:
                    case Function3:
                    default:
                        throw new RuntimeException("Not yet implemented");
                }
            }
        }

        Router router = routingDsl.build();
        int port = 9000;    //TODO don't hard-code...
        Server.forRouter(router, Mode.DEV, port);

    }

    @Override
    public void attach(StatsStorage statsStorage) {

    }

    @Override
    public void detach(StatsStorage statsStorage) {

    }

    @Override
    public boolean isAttached(StatsStorage statsStorage) {
        return false;
    }

    @Override
    public List<StatsStorage> getStatsStorageInstances() {
        return null;
    }
}
