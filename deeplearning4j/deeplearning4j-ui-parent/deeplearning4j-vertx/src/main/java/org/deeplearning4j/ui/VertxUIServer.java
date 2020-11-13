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
import io.vertx.core.Future;
import io.vertx.core.Promise;
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
import org.deeplearning4j.common.config.DL4JClassLoading;
import org.deeplearning4j.common.config.DL4JSystemProperties;
import org.deeplearning4j.common.util.DL4JFileUtils;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.core.storage.StatsStorageEvent;
import org.deeplearning4j.core.storage.StatsStorageListener;
import org.deeplearning4j.core.storage.StatsStorageRouter;
import org.deeplearning4j.exception.DL4JException;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.i18n.I18NProvider;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.model.storage.impl.QueueStatsStorageListener;
import org.deeplearning4j.ui.module.SameDiffModule;
import org.deeplearning4j.ui.module.convolutional.ConvolutionalListenerModule;
import org.deeplearning4j.ui.module.defaultModule.DefaultModule;
import org.deeplearning4j.ui.module.remote.RemoteReceiverModule;
import org.deeplearning4j.ui.module.train.TrainModule;
import org.deeplearning4j.ui.module.tsne.TsneModule;
import org.nd4j.common.function.Function;
import org.nd4j.common.primitives.Pair;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
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
    @Getter
    private static Thread shutdownHook;

    /**
     * Get (and, initialize if necessary) the UI server. This synchronous function will wait until the server started.
     * @param port TCP socket port for {@link HttpServer} to listen
     * @param multiSession         in multi-session mode, multiple training sessions can be visualized in separate browser tabs.
     *                             <br/>URL path will include session ID as a parameter, i.e.: /train becomes /train/:sessionId
     * @param statsStorageProvider function that returns a StatsStorage containing the given session ID.
     *                             <br/>Use this to auto-attach StatsStorage if an unknown session ID is passed
     *                             as URL path parameter in multi-session mode, or leave it {@code null}.
     * @return UI instance for this JVM
     * @throws DL4JException if UI server failed to start;
     * if the instance has already started in a different mode (multi/single-session);
     * if interrupted while waiting for completion
     */
    public static VertxUIServer getInstance(Integer port, boolean multiSession,
                                            Function<String, StatsStorage> statsStorageProvider) throws DL4JException {
        return getInstance(port, multiSession, statsStorageProvider, null);
    }

    /**
     *
     * Get (and, initialize if necessary) the UI server. This function will wait until the server started
     * (synchronous way), or pass the given callback to handle success or failure (asynchronous way).
     * @param port TCP socket port for {@link HttpServer} to listen
     * @param multiSession         in multi-session mode, multiple training sessions can be visualized in separate browser tabs.
     *                             <br/>URL path will include session ID as a parameter, i.e.: /train becomes /train/:sessionId
     * @param statsStorageProvider function that returns a StatsStorage containing the given session ID.
     *                             <br/>Use this to auto-attach StatsStorage if an unknown session ID is passed
     *                             as URL path parameter in multi-session mode, or leave it {@code null}.
     * @param startCallback asynchronous deployment handler callback that will be notify of success or failure.
     *                      If {@code null} given, then this method will wait until deployment is complete.
     *                      If the deployment is successful the result will contain a String representing the
     *                      unique deployment ID of the deployment.
     * @return UI server instance
     * @throws DL4JException if UI server failed to start;
     * if the instance has already started in a different mode (multi/single-session);
     * if interrupted while waiting for completion
     */
    public static VertxUIServer getInstance(Integer port, boolean multiSession,
                                    Function<String, StatsStorage> statsStorageProvider, Promise<String> startCallback)
            throws DL4JException {
        if (instance == null || instance.isStopped()) {
            VertxUIServer.multiSession.set(multiSession);
            VertxUIServer.setStatsStorageProvider(statsStorageProvider);
            instancePort = port;

            if (startCallback != null) {
                //Launch UI server verticle and pass asynchronous callback that will be notified of completion
                deploy(startCallback);
            } else {
                //Launch UI server verticle and wait for it to start
                deploy();
            }
        } else if (!instance.isStopped()) {
            if (multiSession && !instance.isMultiSession()) {
                throw new DL4JException("Cannot return multi-session instance." +
                        " UIServer has already started in single-session mode at " + instance.getAddress() +
                        " You may stop the UI server instance, and start a new one.");
            } else if (!multiSession && instance.isMultiSession()) {
                throw new DL4JException("Cannot return single-session instance." +
                        " UIServer has already started in multi-session mode at " + instance.getAddress() +
                        " You may stop the UI server instance, and start a new one.");
            }
        }

        return instance;
    }

    /**
     * Deploy (start) {@link VertxUIServer}, waiting until starting is complete.
     * @throws DL4JException if UI server failed to start;
     * if interrupted while waiting for completion
     */
    private static void deploy() throws DL4JException {
        CountDownLatch l = new CountDownLatch(1);
        Promise<String> promise = Promise.promise();
        promise.future().compose(
                success -> Future.future(prom -> l.countDown()),
                failure -> Future.future(prom -> l.countDown())
        );
        deploy(promise);
        // synchronous function
        try {
            l.await();
        } catch (InterruptedException e) {
            throw new DL4JException(e);
        }

        Future<String> future = promise.future();
        if (future.failed()) {
            throw new DL4JException("Deeplearning4j UI server failed to start.", future.cause());
        }
    }

    /**
     * Deploy (start) {@link VertxUIServer},
     * and pass callback to handle successful or failed completion of deployment.
     * @param startCallback promise that will handle success or failure of deployment.
     * If the deployment is successful the result will contain a String representing the unique deployment ID of the
     * deployment.
     */
    private static void deploy(Promise<String> startCallback) {
        log.debug("Deeplearning4j UI server is starting.");
        Promise<String> promise = Promise.promise();
        promise.future().compose(
                success -> Future.future(prom -> startCallback.complete(success)),
                failure -> Future.future(prom -> startCallback.fail(new RuntimeException(failure)))
        );

        Vertx vertx = Vertx.vertx();
        vertx.deployVerticle(VertxUIServer.class.getName(), promise);

        VertxUIServer.shutdownHook = new Thread(() -> {
            if (VertxUIServer.instance != null && !VertxUIServer.instance.isStopped()) {
                log.info("Deeplearning4j UI server is auto-stopping in shutdown hook.");
                try {
                    instance.stop();
                } catch (InterruptedException e) {
                    log.error("Interrupted stopping of Deeplearning4j UI server in shutdown hook.", e);
                }
            }
        });
        Runtime.getRuntime().addShutdownHook(shutdownHook);
    }


    private List<UIModule> uiModules = new CopyOnWriteArrayList<>();
    private RemoteReceiverModule remoteReceiverModule;
    /**
     * Loader that attaches {@code StatsStorage} provided by {@code #statsStorageProvider} for the given session ID
     */
    @Getter
    private Function<String, Boolean> statsStorageLoader;

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

    public static void stopInstance() throws Exception {
        if(instance == null || instance.isStopped())
            return;
        instance.stop();
        VertxUIServer.reset();
    }

    private static void reset() {
        VertxUIServer.instance = null;
        VertxUIServer.statsStorageProvider = null;
        VertxUIServer.instancePort = null;
        VertxUIServer.multiSession.set(false);
    }

    /**
     * Auto-attach StatsStorage if an unknown session ID is passed as URL path parameter in multi-session mode
     * @param statsStorageProvider function that returns a StatsStorage containing the given session ID
     */
    public void autoAttachStatsStorageBySessionId(Function<String, StatsStorage> statsStorageProvider) {
        if (statsStorageProvider != null) {
            this.statsStorageLoader = (sessionId) -> {
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
            };
        }
    }

    @Override
    public void start(Promise<Void> startCallback) throws Exception {
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

        if (VertxUIServer.statsStorageProvider != null) {
            autoAttachStatsStorageBySessionId(VertxUIServer.statsStorageProvider);
        }

        uiModules.add(new DefaultModule(isMultiSession())); //For: navigation page "/"
        uiModules.add(new TrainModule());
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

        uiEventRoutingThread = new Thread(new StatsEventRouterRunnable());
        uiEventRoutingThread.setDaemon(true);
        uiEventRoutingThread.start();

        server = vertx.createHttpServer()
                .requestHandler(r)
                .listen(port, result -> {
                    if (result.succeeded()) {
                        String address = UIServer.getInstance().getAddress();
                        log.info("Deeplearning4j UI server started at: {}", address);
                        startCallback.complete();
                    } else {
                        startCallback.fail(new RuntimeException("Deeplearning4j UI server failed to listen on port "
                                + server.actualPort(), result.cause()));
                    }
                });
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
        ServiceLoader<UIModule> sl = DL4JClassLoading.loadService(UIModule.class);
        Iterator<UIModule> iter = sl.iterator();

        if (!iter.hasNext()) {
            return;
        }

        while (iter.hasNext()) {
            UIModule module = iter.next();
            Class<?> moduleClass = module.getClass();
            boolean foundExisting = false;
            for (UIModule mExisting : uiModules) {
                if (mExisting.getClass() == moduleClass) {
                    foundExisting = true;
                    break;
                }
            }

            if (!foundExisting) {
                log.debug("Loaded UI module via service loader: {}", module.getClass());
                uiModules.add(module);
            }
        }
    }

    @Override
    public void stop() throws InterruptedException {
        CountDownLatch l = new CountDownLatch(1);
        Promise<Void> promise = Promise.promise();
        promise.future().compose(
                successEvent -> Future.future(prom -> l.countDown()),
                failureEvent -> Future.future(prom -> l.countDown())
        );
        stopAsync(promise);
        // synchronous function should wait until the server is stopped
        l.await();
    }

    @Override
    public void stopAsync(Promise<Void> stopCallback) {
        /**
         * Stop Vertx instance and release any resources held by it.
         * Pass promise to {@link #stop(Promise)}.
         */
        vertx.close(ar -> stopCallback.handle(ar));
    }

    @Override
    public void stop(Promise<Void> stopCallback) {
        shutdown.set(true);
        stopCallback.complete();
        log.info("Deeplearning4j UI server stopped.");
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
        return "http://localhost:" + server.actualPort();
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
        for (Pair<StatsStorage, StatsStorageListener> p : listeners) {
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
            log.trace("VertxUIServer.StatsEventRouterRunnable started");
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
