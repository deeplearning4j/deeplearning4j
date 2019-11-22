/* ******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.ui;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import io.vertx.core.AbstractVerticle;
import io.vertx.core.Vertx;
import io.vertx.core.http.HttpServer;
import io.vertx.core.http.impl.MimeMapping;
import io.vertx.ext.web.Router;
import io.vertx.ext.web.RoutingContext;
import io.vertx.ext.web.handler.BodyHandler;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.api.storage.StatsStorageListener;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.config.DL4JSystemProperties;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.i18n.I18NProvider;
import org.deeplearning4j.ui.module.SameDiffModule;
import org.deeplearning4j.ui.module.convolutional.ConvolutionalListenerModule;
import org.deeplearning4j.ui.module.defaultModule.DefaultModule;
import org.deeplearning4j.ui.module.remote.RemoteReceiverModule;
import org.deeplearning4j.ui.module.train.TrainModule;
import org.deeplearning4j.ui.module.tsne.TsneModule;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.storage.impl.QueueStatsStorageListener;
import org.deeplearning4j.util.DL4JFileUtils;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

@Slf4j
public class VertxUIServer extends AbstractVerticle implements UIServer {
    public static final int DEFAULT_UI_PORT = 9000;
    public static final String ASSETS_ROOT_DIRECTORY = "deeplearning4jUiAssets/";

    @Getter
    private static VertxUIServer instance;

    @Getter
    private static AtomicBoolean multiSession = new AtomicBoolean(false);
    @Getter
    @Setter
    private static Function<String, StatsStorage> statsStorageProvider;

    private static Integer instancePort;

    public static VertxUIServer getInstance() {
        return getInstance(null, multiSession.get(), null);
    }

    public static VertxUIServer getInstance(Integer port, boolean multiSession, Function<String, StatsStorage> statsStorageProvider){
        if (instance == null || instance.isStopped()) {
            VertxUIServer.multiSession.set(multiSession);
            VertxUIServer.setStatsStorageProvider(statsStorageProvider);
            instancePort = port;
            Vertx vertx = Vertx.vertx();

            //Launch UI server verticle and wait for it to start
            CountDownLatch l = new CountDownLatch(1);
            vertx.deployVerticle(VertxUIServer.class.getName(), res -> {
                l.countDown();
            });
            try {
                l.await(5000, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e){ } //Ignore
        } else if (!instance.isStopped()) {
            if (instance.multiSession.get() && !instance.isMultiSession()) {
                throw new RuntimeException("Cannot return multi-session instance." +
                        " UIServer has already started in single-session mode at " + instance.getAddress() +
                        " You may stop the UI server instance, and start a new one.");
            } else if (!instance.multiSession.get() && instance.isMultiSession()) {
                throw new RuntimeException("Cannot return single-session instance." +
                        " UIServer has already started in multi-session mode at " + instance.getAddress() +
                        " You may stop the UI server instance, and start a new one.");
            }
        }

        return instance;
    }


    private List<UIModule> uiModules = new CopyOnWriteArrayList<>();
    private RemoteReceiverModule remoteReceiverModule;
    private StatsStorageLoader statsStorageLoader;

    //typeIDModuleMap: Records which modules are registered for which type IDs
    private Map<String, List<UIModule>> typeIDModuleMap = new ConcurrentHashMap<>();

    private HttpServer server;
    private AtomicBoolean shutdown = new AtomicBoolean(false);
    private long uiProcessingDelay = 500; //500ms. TODO make configurable


    private final BlockingQueue<StatsStorageEvent> eventQueue = new LinkedBlockingQueue<>();
    private List<Pair<StatsStorage, StatsStorageListener>> listeners = new CopyOnWriteArrayList<>();
    private List<StatsStorage> statsStorageInstances = new CopyOnWriteArrayList<>();

    private Thread uiEventRoutingThread;

    public VertxUIServer() {
        instance = this;
    }

    public static void stopInstance(){
        if(instance == null)
            return;
        instance.stop();
    }

    @Override
    public void start() throws Exception {
        //Create REST endpoints
        File uploadDir = new File(System.getProperty("java.io.tmpdir"), "DL4JUI_" + System.currentTimeMillis());
        uploadDir.mkdirs();
        Router r = Router.router(vertx);
        r.route().handler(BodyHandler.create()  //NOTE: Setting this is required to receive request body content at all
                .setUploadsDirectory(uploadDir.getAbsolutePath()));
        r.get("/assets/*").handler(rc -> {
            String path = rc.request().path();
            path = path.substring(8);   //Remove "/assets/", which is 8 characters
            String mime;
            String newPath;
            if (path.contains("webjars")) {
                newPath = "META-INF/resources/" + path.substring(path.indexOf("webjars"));
            } else {
                newPath = ASSETS_ROOT_DIRECTORY + (path.startsWith("/") ? path.substring(1) : path);
            }
            mime = MimeMapping.getMimeTypeForFilename(FilenameUtils.getName(newPath));

            //System.out.println("PATH: " + path + " - mime = " + mime);
            rc.response()
                    .putHeader("content-type", mime)
                    .sendFile(newPath);
        });


        if (isMultiSession()) {
            r.get("/setlang/:sessionId/:to").handler(
                    rc -> {
                        String sid = rc.request().getParam("sessionID");
                        String to = rc.request().getParam("to");
                        I18NProvider.getInstance(sid).setDefaultLanguage(to);
                        rc.response().end();
                    });
        } else {
            r.get("/setlang/:to").handler(rc -> {
                String to = rc.request().getParam("to");
                I18NProvider.getInstance().setDefaultLanguage(to);
                rc.response().end();
            });
        }


        uiModules.add(new DefaultModule(isMultiSession())); //For: navigation page "/"
        uiModules.add(new TrainModule(isMultiSession(), statsStorageLoader, this::getAddress));
        uiModules.add(new ConvolutionalListenerModule());
        uiModules.add(new TsneModule());
        uiModules.add(new SameDiffModule());
        remoteReceiverModule = new RemoteReceiverModule();
        uiModules.add(remoteReceiverModule);

        //Check service loader mechanism (Arbiter UI, etc) for modules
        modulesViaServiceLoader(uiModules);

        for (UIModule m : uiModules) {
            List<Route> routes = m.getRoutes();
            for (Route route : routes) {
                switch (route.getHttpMethod()) {
                    case GET:
                        r.get(route.getRoute()).handler(rc -> route.getConsumer().accept(extractArgsFromRoute(route.getRoute(), rc), rc));
                        break;
                    case PUT:
                        r.put(route.getRoute()).handler(rc -> route.getConsumer().accept(extractArgsFromRoute(route.getRoute(), rc), rc));
                        break;
                    case POST:
                        r.post(route.getRoute()).handler(rc -> route.getConsumer().accept(extractArgsFromRoute(route.getRoute(), rc), rc));
                        break;
                    default:
                        throw new IllegalStateException("Unknown or not supported HTTP method: " + route.getHttpMethod());
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

        //Check port property
        int port = instancePort == null ? DEFAULT_UI_PORT : instancePort;
        String portProp = System.getenv(DL4JSystemProperties.UI_SERVER_PORT_PROPERTY);
        if(portProp != null && !portProp.isEmpty()){
            try{
                port = Integer.parseInt(portProp);
            } catch (NumberFormatException e){
                log.warn("Error parsing port property {}={}", DL4JSystemProperties.UI_SERVER_PORT_PROPERTY, portProp);
            }
        }


        server = vertx.createHttpServer()
                .requestHandler(r)
                .listen(port);

        uiEventRoutingThread = new Thread(new StatsEventRouterRunnable());
        uiEventRoutingThread.setDaemon(true);
        uiEventRoutingThread.start();
    }

    private List<String> extractArgsFromRoute(String path, RoutingContext rc) {
        if (!path.contains(":")) {
            return Collections.emptyList();
        }
        String[] split = path.split("/");
        List<String> out = new ArrayList<>();
        for (String s : split) {
            if (s.startsWith(":")) {
                String s2 = s.substring(1);
                out.add(rc.request().getParam(s2));
            }
        }
        return out;
    }

    private void modulesViaServiceLoader(List<UIModule> uiModules) {

        ServiceLoader<UIModule> sl = ServiceLoader.load(UIModule.class);
        Iterator<UIModule> iter = sl.iterator();

        if (!iter.hasNext()) {
            return;
        }

        while (iter.hasNext()) {
            UIModule m = iter.next();
            Class<?> c = m.getClass();
            boolean foundExisting = false;
            for (UIModule mExisting : uiModules) {
                if (mExisting.getClass() == c) {
                    foundExisting = true;
                    break;
                }
            }

            if (!foundExisting) {
                log.debug("Loaded UI module via service loader: {}", m.getClass());
                uiModules.add(m);
            }
        }
    }

    @Override
    public void stop() {
        server.close();
        shutdown.set(true);
    }


    @Override
    public boolean isStopped() {
        return shutdown.get();
    }

    @Override
    public boolean isMultiSession() {
        return multiSession.get();
    }

    @Override
    public String getAddress() {
        return "https://localhost:" + server.actualPort();
    }

    @Override
    public int getPort() {
        return server.actualPort();
    }

    @Override
    public void attach(StatsStorage statsStorage) {
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
    public void detach(StatsStorage statsStorage) {
        if (statsStorage == null)
            throw new IllegalArgumentException("StatsStorage cannot be null");
        if (!statsStorageInstances.contains(statsStorage))
            return; //No op
        boolean found = false;
        for (Iterator<Pair<StatsStorage, StatsStorageListener>> iterator = listeners.iterator(); iterator.hasNext(); ) {
            Pair<StatsStorage, StatsStorageListener> p = iterator.next();
            if (p.getFirst() == statsStorage) { //Same object, not equality
                statsStorage.deregisterStatsStorageListener(p.getSecond());
                listeners.remove(p);
                found = true;
            }
        }
        statsStorageInstances.remove(statsStorage);
        for (UIModule uiModule : uiModules) {
            uiModule.onDetach(statsStorage);
        }
        for (String sessionId : statsStorage.listSessionIDs()) {
            I18NProvider.removeInstance(sessionId);
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
            log.info("VertxUIServer.StatsEventRouterRunnable started");
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
                        if (callbackTypes.contains(e.getTypeID())
                                && statsStorageInstances.contains(e.getStatsStorage())) {
                            out.add(e);
                        }
                    }

                    m.reportStorageEvents(out);
                }

                events.clear();

                try {
                    Thread.sleep(uiProcessingDelay);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    if (!shutdown.get()) {
                        throw new RuntimeException("Unexpected interrupted exception", e);
                    }
                }
            }
        }
    }

    /**
     * Loader that attaches {@code StatsStorage} provided by {@code #statsStorageProvider} for the given session ID
     */
    private class StatsStorageLoader implements Function<String, Boolean> {

        Function<String, StatsStorage> statsStorageProvider;

        StatsStorageLoader(Function<String, StatsStorage> statsStorageProvider) {
            this.statsStorageProvider = statsStorageProvider;
        }

        @Override
        public Boolean apply(String sessionId) {
            log.info("Loading StatsStorage via StatsStorageProvider for session ID (" + sessionId + ").");
            StatsStorage statsStorage = statsStorageProvider.apply(sessionId);
            if (statsStorage != null) {
                if (statsStorage.sessionExists(sessionId)) {
                    attach(statsStorage);
                    return true;
                }
                log.info("Failed to load StatsStorage via StatsStorageProvider for session ID. " +
                        "Session ID (" + sessionId + ") does not exist in StatsStorage.");
                return false;
            } else {
                log.info("Failed to load StatsStorage via StatsStorageProvider for session ID (" + sessionId + "). " +
                        "StatsStorageProvider returned null.");
                return false;
            }
        }
    }

    //==================================================================================================================
    // CLI Launcher

    @Data
    private static class CLIParams {
        @Parameter(names = {"-r", "--enableRemote"}, description = "Whether to enable remote or not", arity = 1)
        private boolean cliEnableRemote;

        @Parameter(names = {"-p", "--uiPort"}, description = "Custom HTTP port for UI", arity = 1)
        private int cliPort = DEFAULT_UI_PORT;

        @Parameter(names = {"-f", "--customStatsFile"}, description = "Path to create custom stats file (remote only)", arity = 1)
        private String cliCustomStatsFile;

        @Parameter(names = {"-m", "--multiSession"}, description = "Whether to enable multiple separate browser sessions or not", arity = 1)
        private boolean cliMultiSession;
    }

    public void main(String[] args){
        CLIParams d = new CLIParams();
        new JCommander(d).parse(args);
        instancePort = d.getCliPort();
        UIServer.getInstance(d.isCliMultiSession(), null);
        if(d.isCliEnableRemote()){
            try {
                File tempStatsFile = DL4JFileUtils.createTempFile("dl4j", "UIstats");
                tempStatsFile.delete();
                tempStatsFile.deleteOnExit();
                enableRemoteListener(new FileStatsStorage(tempStatsFile), true);
            } catch(Exception e) {
                log.error("Failed to create temporary file for stats storage",e);
                System.exit(1);
            }
        }
    }
}
