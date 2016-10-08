package org.deeplearning4j.ui.play;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.modules.histogram.HistogramModule;
import org.deeplearning4j.ui.storage.StatsStorage;
import org.deeplearning4j.ui.storage.StatsStorageEvent;
import org.deeplearning4j.ui.storage.StatsStorageListener;
import org.deeplearning4j.ui.storage.impl.QueuePairStatsStorageListener;
import org.deeplearning4j.ui.storage.impl.QueueStatsStorageListener;
import org.nd4j.linalg.util.MultiValueMap;
import play.Mode;
import play.mvc.Result;
import play.routing.Router;
import play.routing.RoutingDsl;
import play.server.Server;

import java.util.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

import static play.mvc.Results.ok;

/**
 * A UI server based on the Play framework
 *
 * @author Alex Black
 */
@Slf4j
public class PlayUIServer extends UIServer {

    private Server server;
    private final BlockingQueue<Pair<StatsStorage, StatsStorageEvent>> eventQueue = new LinkedBlockingQueue<>();
    private List<Pair<StatsStorage, StatsStorageListener>> listeners = new ArrayList<>();
    private List<StatsStorage> statsStorageInstances = new ArrayList<>();

    private List<UIModule> uiModules = new ArrayList<>();
    //typeIDModuleMap: Records which modules are registered for which type IDs
    private Map<String, List<UIModule>> typeIDModuleMap = new ConcurrentHashMap<>();

    private long uiProcessingDelay = 500;   //500ms. TODO make configurable
    private final AtomicBoolean shutdown = new AtomicBoolean(false);

    public PlayUIServer() {

        RoutingDsl routingDsl = new RoutingDsl();

        routingDsl.GET("/").routeTo(new Index());

        uiModules.add(new HistogramModule());       //TODO don't hardcode and/or add reflection...

        for (UIModule m : uiModules) {
            List<Route> routes = m.getRoutes();
            for (Route r : routes) {
                RoutingDsl.PathPatternMatcher ppm = routingDsl.match(r.getHttpMethod().name(), r.getRoute());
                switch (r.getFunctionType()) {
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

            //Determine which type IDs this module wants to receive:
            List<String> typeIDs = m.getCallbackTypeIDs();
            for (String typeID : typeIDs) {
                List<UIModule> list = typeIDModuleMap.get(typeID);
                if (list == null) {
                    list = Collections.synchronizedList(new ArrayList<>());
                    typeIDModuleMap.put(typeID, list);
                }
                list.add(m);
            }
        }


        Router router = routingDsl.build();
        int port = 9000;    //TODO don't hard-code...
        server = Server.forRouter(router, Mode.DEV, port);

        log.info("UI Server started at {}", server.mainAddress());
    }

    @Override
    public synchronized void attach(StatsStorage statsStorage) {
        if (statsStorage == null) throw new IllegalArgumentException("StatsStorage cannot be null");
        StatsStorageListener listener = new QueuePairStatsStorageListener(statsStorage, eventQueue);
        listeners.add(new Pair<>(statsStorage, listener));
        statsStorage.registerStatsStorageListener(listener);
        statsStorageInstances.add(statsStorage);
        log.info("StatsStorage instance attached to UI: {}", statsStorage);
    }

    @Override
    public synchronized void detach(StatsStorage statsStorage) {
        if (statsStorage == null) throw new IllegalArgumentException("StatsStorage cannot be null");
        boolean found = false;
        for (Pair<StatsStorage, StatsStorageListener> p : listeners) {
            if (p.getFirst() == statsStorage) {       //Same object, not equality
                statsStorage.deregisterStatsStorageListener(p.getSecond());
                listeners.remove(p);
                found = true;
            }
        }
        if (found) {
            log.info("StatsStorage instance detached from UI: {}", statsStorage);
        }
    }

    @Override
    public boolean isAttached(StatsStorage statsStorage) {
        return statsStorageInstances.contains(statsStorage);
    }

    @Override
    public List<StatsStorage> getStatsStorageInstances() {
        return new ArrayList<>(statsStorageInstances);
    }


    private class StatsEventRouterRunnable implements Runnable {


        @Override
        public void run() {
            try {
                runHelper();
            } catch (Exception e) {
                log.error("Unexpected exception from Event routing runnable", e);
            }
        }

        private void runHelper() throws Exception {
            //Idea: collect all event stats, and route them to the appropriate modules
            while (!shutdown.get()) {

                List<Pair<StatsStorage,StatsStorageEvent>> events = new ArrayList<>();
                Pair<StatsStorage,StatsStorageEvent> sse = eventQueue.take();  //Blocking operation
                events.add(sse);
                eventQueue.drainTo(events); //Non-blocking

                //First: group by StatsStorage
                Map<StatsStorage, List<StatsStorageEvent>> eventsBySource = new HashMap<>();
                for(Pair<StatsStorage, StatsStorageEvent> p : events){
                    List<StatsStorageEvent> list = eventsBySource.get(p.getFirst());
                    if(list == null){
                        list = new ArrayList<>();
                        eventsBySource.put(p.getFirst(), list);
                    }
                    list.add(p.getSecond());
                }

                //Second: for each StatsStorage instance, sort by UI module and route to the appropriate locations...
                int count = 0;
                int skipped = 0;
                for(Map.Entry<StatsStorage, List<StatsStorageEvent>> entry : eventsBySource.entrySet()){

                    Map<UIModule, List<StatsStorageEvent>> eventsByModule = new HashMap<>();
                    for (Pair<StatsStorage,StatsStorageEvent> event : events) {
                        String typeID = event.getSecond().getTypeID();
                        if (!typeIDModuleMap.containsKey(typeID)) {
                            skipped++;
                            continue;
                        }

                        List<UIModule> moduleList = typeIDModuleMap.get(typeID);
                        for (UIModule m : moduleList) {
                            List<StatsStorageEvent> eventsForModule = eventsByModule.get(m);
                            if (eventsForModule == null) {
                                eventsForModule = new ArrayList<>();
                                eventsByModule.put(m, eventsForModule);
                            }
                            eventsForModule.add(event.getSecond());
                        }
                        count++;
                    }

                    //Actually report to the appropriate modules
                    for (Map.Entry<UIModule, List<StatsStorageEvent>> entryModule : eventsByModule.entrySet()) {
                        entryModule.getKey().reportStorageEvents(entry.getKey(), entryModule.getValue());
                    }
                }


                log.debug("Reported {} events to UI modules with {} skipped", count, skipped);

                try {
                    Thread.sleep(uiProcessingDelay);
                } catch (InterruptedException e) {
                    if (!shutdown.get()) {
                        throw new RuntimeException("Unexpected interrupted exception", e);
                    }
                }
            }

        }
    }
}
