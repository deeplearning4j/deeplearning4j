package org.deeplearning4j.ui.play;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.i18n.I18NProvider;
import org.deeplearning4j.ui.module.convolutional.ConvolutionalListenerModule;
import org.deeplearning4j.ui.module.defaultModule.DefaultModule;
import org.deeplearning4j.ui.module.flow.FlowListenerModule;
import org.deeplearning4j.ui.module.train.TrainModule;
import org.deeplearning4j.ui.module.histogram.HistogramModule;
import org.deeplearning4j.ui.module.tsne.TsneModule;
import org.deeplearning4j.ui.play.misc.FunctionUtil;
import org.deeplearning4j.ui.play.staticroutes.Assets;
import org.deeplearning4j.ui.play.staticroutes.I18NRoute;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.api.storage.StatsStorageListener;
import org.deeplearning4j.ui.storage.impl.QueuePairStatsStorageListener;
import org.deeplearning4j.ui.storage.impl.QueueStatsStorageListener;
import play.Mode;
import play.api.routing.Router;
import play.routing.RoutingDsl;
import play.server.Server;

import java.util.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;

import static play.mvc.Results.ok;

/**
 * A UI server based on the Play framework
 *
 * @author Alex Black
 */
@Slf4j
public class PlayUIServer extends UIServer {

    /**
     * System property for setting the UI port. Defaults to 9000.
     * Set to 0 to use a random port
     */
    public static final String UI_SERVER_PORT_PROPERTY = "org.deeplearning4j.ui.port";
    public static final int DEFAULT_UI_PORT = 9000;

    public static final String ASSETS_ROOT_DIRECTORY = "deeplearning4jUiAssets/";

    private Server server;
    private final BlockingQueue<StatsStorageEvent> eventQueue = new LinkedBlockingQueue<>();
    private List<Pair<StatsStorage, StatsStorageListener>> listeners = new ArrayList<>();
    private List<StatsStorage> statsStorageInstances = new ArrayList<>();

    private List<UIModule> uiModules = new ArrayList<>();
    //typeIDModuleMap: Records which modules are registered for which type IDs
    private Map<String, List<UIModule>> typeIDModuleMap = new ConcurrentHashMap<>();

    private long uiProcessingDelay = 500;   //500ms. TODO make configurable
    private final AtomicBoolean shutdown = new AtomicBoolean(false);

    private Thread uiEventRoutingThread;

    private int port;

    public PlayUIServer() {

        RoutingDsl routingDsl = new RoutingDsl();

        //Set up index page and assets routing
        //The definitions and FunctionUtil may look a bit weird here... this is used to translate implementation independent
        // definitions (i.e., Java Supplier, Function etc interfaces) to the Play-specific versions
        //This way, routing is not directly dependent ot Play API. Furthermore, Play 2.5 switches to using these Java interfaces
        // anyway; thus switching 2.5 should be as simple as removing the FunctionUtil calls...
//        routingDsl.GET("/").routeTo(FunctionUtil.function0(new Index()));
        routingDsl.GET("/setlang/:to").routeTo(FunctionUtil.function(new I18NRoute()));
        routingDsl.GET("/lang/getCurrent").routeTo(() -> ok(I18NProvider.getInstance().getDefaultLanguage()));
        routingDsl.GET("/assets/*file").routeTo(FunctionUtil.function(new Assets(ASSETS_ROOT_DIRECTORY)));

        uiModules.add(new DefaultModule());         //For: navigation page "/"
        uiModules.add(new HistogramModule());       //TODO don't hardcode and/or add reflection...
        uiModules.add(new TrainModule());
        uiModules.add(new ConvolutionalListenerModule());
        uiModules.add(new FlowListenerModule());
        uiModules.add(new TsneModule());


        for (UIModule m : uiModules) {
            List<Route> routes = m.getRoutes();
            for (Route r : routes) {
                RoutingDsl.PathPatternMatcher ppm = routingDsl.match(r.getHttpMethod().name(), r.getRoute());
                switch (r.getFunctionType()) {
                    case Supplier:
                        ppm.routeTo(FunctionUtil.function0(r.getSupplier()));
                        break;
                    case Function:
                        ppm.routeTo(FunctionUtil.function(r.getFunction()));
                        break;
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

        String portProperty = System.getProperty(UI_SERVER_PORT_PROPERTY);
        int port = DEFAULT_UI_PORT;
        if(portProperty != null){
            try{
                port = Integer.parseInt(portProperty);
            }catch(NumberFormatException e){
                log.warn("Could not parse {} property: NumberFormatException for property value \"{}\". Defaulting to port {}. Set property to 0 for random port",
                        UI_SERVER_PORT_PROPERTY, portProperty, port);
            }
        }

        Router router = routingDsl.build();
        server = Server.forRouter(router, Mode.DEV, port);
        this.port = port;

        String addr = server.mainAddress().toString();
        if(addr.startsWith("/0:0:0:0:0:0:0:0")){
            int last = addr.lastIndexOf(':');
            if(last > 0) {
                addr = "http://localhost:" + addr.substring(last+1);
            }
        }
        log.info("UI Server started at {}", addr);

        uiEventRoutingThread = new Thread(new StatsEventRouterRunnable());
        uiEventRoutingThread.setDaemon(true);
        uiEventRoutingThread.start();
    }

    @Override
    public int getPort() {
        return port;
    }

    @Override
    public synchronized void attach(StatsStorage statsStorage) {
        if (statsStorage == null) throw new IllegalArgumentException("StatsStorage cannot be null");
        if (statsStorageInstances.contains(statsStorage)) return;
        StatsStorageListener listener = new QueueStatsStorageListener(eventQueue);
        listeners.add(new Pair<>(statsStorage, listener));
        statsStorage.registerStatsStorageListener(listener);
        statsStorageInstances.add(statsStorage);

        for (UIModule uiModule : uiModules) {
            uiModule.onAttach(statsStorage);
        }

        log.info("StatsStorage instance attached to UI: {}", statsStorage);
    }

    @Override
    public synchronized void detach(StatsStorage statsStorage) {
        if (statsStorage == null) throw new IllegalArgumentException("StatsStorage cannot be null");
        if (!statsStorageInstances.contains(statsStorage)) return;   //No op
        boolean found = false;
        for (Pair<StatsStorage, StatsStorageListener> p : listeners) {
            if (p.getFirst() == statsStorage) {       //Same object, not equality
                statsStorage.deregisterStatsStorageListener(p.getSecond());
                listeners.remove(p);
                found = true;
            }
        }
        for (UIModule uiModule : uiModules) {
            uiModule.onDetach(statsStorage);
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
            log.info("PlayUIServer.StatsEventRouterRunnable started");
            //Idea: collect all event stats, and route them to the appropriate modules
            while (!shutdown.get()) {

                List<StatsStorageEvent> events = new ArrayList<>();
                StatsStorageEvent sse = eventQueue.take();  //Blocking operation
                events.add(sse);
                eventQueue.drainTo(events); //Non-blocking

                for(UIModule m : uiModules){

                    List<String> callbackTypes = m.getCallbackTypeIDs();
                    List<StatsStorageEvent> out = new ArrayList<>();
                    for(StatsStorageEvent e : events){
                        if(callbackTypes.contains(e.getTypeID())){
                            out.add(e);
                        }
                    }

                    m.reportStorageEvents(out);
                }

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
