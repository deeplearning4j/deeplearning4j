package org.deeplearning4j.ui.play;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import com.google.common.collect.Sets;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.api.storage.StatsStorageListener;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.i18n.I18NProvider;
import org.deeplearning4j.ui.module.convolutional.ConvolutionalListenerModule;
import org.deeplearning4j.ui.module.defaultModule.DefaultModule;
import org.deeplearning4j.ui.module.remote.RemoteReceiverModule;
import org.deeplearning4j.ui.module.train.TrainModule;
import org.deeplearning4j.ui.module.tsne.TsneModule;
import org.deeplearning4j.ui.play.misc.FunctionUtil;
import org.deeplearning4j.ui.play.staticroutes.Assets;
import org.deeplearning4j.ui.play.staticroutes.I18NRoute;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.storage.impl.QueueStatsStorageListener;
import org.reflections.ReflectionUtils;
import org.reflections.Reflections;
import play.Mode;
import play.api.routing.Router;
import play.routing.RoutingDsl;
import play.server.Server;

import java.util.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;

import static play.mvc.Results.ok;

/**
 * A UI server based on the Play framework
 *
 * @author Alex Black
 */
@Slf4j
@Data
public class PlayUIServer extends UIServer {

    /**
     * System property for setting the UI port. Defaults to 9000.
     * Set to 0 to use a random port
     */
    public static final String UI_SERVER_PORT_PROPERTY = "org.deeplearning4j.ui.port";
    public static final int DEFAULT_UI_PORT = 9000;

    /**
     * System property to enable classpath scanning for custom UI modules. Disabled by default.
     */
    public static final String UI_CUSTOM_MODULE_PROPERTY = "org.deeplearning4j.ui.custommodule.enable";


    public static final String ASSETS_ROOT_DIRECTORY = "deeplearning4jUiAssets/";

    private Server server;
    private final BlockingQueue<StatsStorageEvent> eventQueue = new LinkedBlockingQueue<>();
    private List<Pair<StatsStorage, StatsStorageListener>> listeners = new CopyOnWriteArrayList<>();
    private List<StatsStorage> statsStorageInstances = new CopyOnWriteArrayList<>();

    private List<UIModule> uiModules = new CopyOnWriteArrayList<>();
    private RemoteReceiverModule remoteReceiverModule;
    //typeIDModuleMap: Records which modules are registered for which type IDs
    private Map<String, List<UIModule>> typeIDModuleMap = new ConcurrentHashMap<>();

    private long uiProcessingDelay = 500; //500ms. TODO make configurable
    private final AtomicBoolean shutdown = new AtomicBoolean(false);

    private Thread uiEventRoutingThread;
    @Parameter(names = {"-r", "-enableRemote"}, description = "Whether to enable remote or not", arity = 1)
    private boolean enableRemote;


    @Parameter(names = {"--uiPort"}, description = "Whether to enable remote or not", arity = 1)
    private int port = DEFAULT_UI_PORT;

    public PlayUIServer() {
        this(DEFAULT_UI_PORT);
    }

    public PlayUIServer(int port) {
        this.port = port;
    }


    public void runMain(String[] args) {
        JCommander jcmdr = new JCommander(this);

        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            System.exit(1);
        }
        RoutingDsl routingDsl = new RoutingDsl();

        //Set up index page and assets routing
        //The definitions and FunctionUtil may look a bit weird here... this is used to translate implementation independent
        // definitions (i.e., Java Supplier, Function etc interfaces) to the Play-specific versions
        //This way, routing is not directly dependent ot Play API. Furthermore, Play 2.5 switches to using these Java interfaces
        // anyway; thus switching 2.5 should be as simple as removing the FunctionUtil calls...
        routingDsl.GET("/setlang/:to").routeTo(FunctionUtil.function(new I18NRoute()));
        routingDsl.GET("/lang/getCurrent").routeTo(() -> ok(I18NProvider.getInstance().getDefaultLanguage()));
        routingDsl.GET("/assets/*file").routeTo(FunctionUtil.function(new Assets(ASSETS_ROOT_DIRECTORY)));

        uiModules.add(new DefaultModule()); //For: navigation page "/"
        uiModules.add(new TrainModule());
        uiModules.add(new ConvolutionalListenerModule());
        uiModules.add(new TsneModule());
        remoteReceiverModule = new RemoteReceiverModule();
        uiModules.add(remoteReceiverModule);

        //Check service loader mechanism (Arbiter UI, etc) for modules
        uiModules.addAll(modulesViaServiceLoader());


        //Check if custom UI modules are enabled...
        String customModulePropertyStr = System.getProperty(UI_CUSTOM_MODULE_PROPERTY);
        boolean useCustomModules = false;
        if (customModulePropertyStr != null) {
            useCustomModules = Boolean.parseBoolean(customModulePropertyStr);
        }

        if (useCustomModules) {
            List<Class<?>> excludeClasses = new ArrayList<>();
            for (UIModule u : uiModules) {
                excludeClasses.add(u.getClass());
            }
            List<UIModule> list = getCustomUIModules(excludeClasses);
            uiModules.addAll(list);
        }



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
        if (portProperty != null) {
            try {
                port = Integer.parseInt(portProperty);
            } catch (NumberFormatException e) {
                log.warn("Could not parse UI port property \"{}\" with value \"{}\"", UI_SERVER_PORT_PROPERTY,
                                portProperty, e);
            }
        }

        Router router = routingDsl.build();
        server = Server.forRouter(router, Mode.DEV, port);
        this.port = port;

        log.info("DL4J UI Server started at {}", getAddress());

        uiEventRoutingThread = new Thread(new StatsEventRouterRunnable());
        uiEventRoutingThread.setDaemon(true);
        uiEventRoutingThread.start();
        if (enableRemote)
            enableRemoteListener();
    }

    @Override
    public String getAddress() {
        String addr = server.mainAddress().toString();
        if (addr.startsWith("/0:0:0:0:0:0:0:0")) {
            int last = addr.lastIndexOf(':');
            if (last > 0) {
                addr = "http://localhost:" + addr.substring(last + 1);
            }
        }
        return addr;
    }

    private List<UIModule> modulesViaServiceLoader() {

        ServiceLoader<UIModule> sl = ServiceLoader.load(UIModule.class);
        Iterator<UIModule> iter = sl.iterator();

        if (!iter.hasNext()) {
            return Collections.emptyList();
        }

        List<UIModule> l = new ArrayList<>();
        while (iter.hasNext()) {
            UIModule m = iter.next();
            log.debug("Loaded UI module via service loader: {}", m.getClass());
            l.add(m);
        }

        return l;
    }


    public static void main(String[] args) {
        new PlayUIServer().runMain(args);
    }

    private List<UIModule> getCustomUIModules(List<Class<?>> excludeClasses) {
        //Scan classpath for UI module instances, but ignore the 'excludeClasses' classes
        List<String> classNames = Collections.singletonList(UIModule.class.getName());
        Reflections reflections = new Reflections();
        org.reflections.Store store = reflections.getStore();
        Iterable<String> subtypesByName =
                        store.getAll(org.reflections.scanners.SubTypesScanner.class.getSimpleName(), classNames);
        Set<? extends Class<?>> subtypeClasses = Sets.newHashSet(ReflectionUtils.forNames(subtypesByName));

        List<Class<?>> toCreate = new ArrayList<>();
        for (Class<?> c : subtypeClasses) {
            if (excludeClasses.contains(c))
                continue;;
            toCreate.add(c);
        }

        List<UIModule> ret = new ArrayList<>(toCreate.size());
        for (Class<?> c : toCreate) {
            UIModule m;
            try {
                m = (UIModule) c.newInstance();
            } catch (Exception e) {
                log.warn("Could not create instance of custom UIModule of type {}; skipping", c, e);
                continue;
            }
            log.debug("Created instance of custom UI module: {}", c);
            ret.add(m);
        }

        return ret;
    }

    @Override
    public int getPort() {
        return port;
    }

    @Override
    public synchronized void attach(StatsStorage statsStorage) {
        if (statsStorage == null)
            throw new IllegalArgumentException("StatsStorage cannot be null");
        if (statsStorageInstances.contains(statsStorage))
            return;
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
        if (statsStorage == null)
            throw new IllegalArgumentException("StatsStorage cannot be null");
        if (!statsStorageInstances.contains(statsStorage))
            return; //No op
        boolean found = false;
        for (Iterator<Pair<StatsStorage, StatsStorageListener>> iterator = listeners.iterator(); iterator.hasNext();) {
            Pair<StatsStorage, StatsStorageListener> p = iterator.next();
            if (p.getFirst() == statsStorage) { //Same object, not equality
                statsStorage.deregisterStatsStorageListener(p.getSecond());
                iterator.remove();
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

    @Override
    public void enableRemoteListener() {
        if (remoteReceiverModule == null)
            remoteReceiverModule = new RemoteReceiverModule();
        if (remoteReceiverModule.isEnabled())
            return;
        enableRemoteListener(new InMemoryStatsStorage(), true);
    }

    @Override
    public void enableRemoteListener(StatsStorageRouter statsStorage, boolean attach) {
        remoteReceiverModule.setEnabled(true);
        remoteReceiverModule.setStatsStorage(statsStorage);
        if (attach && statsStorage instanceof StatsStorage) {
            attach((StatsStorage) statsStorage);
        }
    }

    @Override
    public void disableRemoteListener() {
        remoteReceiverModule.setEnabled(false);
    }

    @Override
    public boolean isRemoteListenerEnabled() {
        return remoteReceiverModule.isEnabled();
    }

    @Override
    public void stop() {
        if (server != null)
            server.stop();
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
            log.debug("PlayUIServer.StatsEventRouterRunnable started");
            //Idea: collect all event stats, and route them to the appropriate modules
            while (!shutdown.get()) {

                List<StatsStorageEvent> events = new ArrayList<>();
                StatsStorageEvent sse = eventQueue.take(); //Blocking operation
                events.add(sse);
                eventQueue.drainTo(events); //Non-blocking

                for (UIModule m : uiModules) {

                    List<String> callbackTypes = m.getCallbackTypeIDs();
                    List<StatsStorageEvent> out = new ArrayList<>();
                    for (StatsStorageEvent e : events) {
                        if (callbackTypes.contains(e.getTypeID())) {
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
