/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.factory;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.config.ND4JClassLoading;
import org.nd4j.common.config.ND4JEnvironmentVars;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.context.Nd4jContext;
import org.nd4j.common.io.Resource;

import java.io.File;
import java.io.IOException;
import java.net.URLClassLoader;
import java.security.PrivilegedActionException;
import java.util.*;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.ConcurrentHashMap;

@Slf4j
public abstract class Nd4jBackend {

    public static final int BACKEND_PRIORITY_CPU;
    public static final int BACKEND_PRIORITY_GPU;

    /**
     * @deprecated Use {@link ND4JEnvironmentVars#BACKEND_DYNAMIC_LOAD_CLASSPATH}
     */
    @Deprecated
    public final static String DYNAMIC_LOAD_CLASSPATH = ND4JEnvironmentVars.BACKEND_DYNAMIC_LOAD_CLASSPATH;
    /**
     * @deprecated Use {@link ND4JSystemProperties#DYNAMIC_LOAD_CLASSPATH_PROPERTY}
     */
    @Deprecated
    public final static String DYNAMIC_LOAD_CLASSPATH_PROPERTY = ND4JSystemProperties.DYNAMIC_LOAD_CLASSPATH_PROPERTY;

    // Thread-safe tracking of dynamic loading attempts
    private static final AtomicBoolean triedDynamicLoad = new AtomicBoolean(false);

    // Thread-safe tracking of backend loading state
    private static final AtomicBoolean isLoaded = new AtomicBoolean(false);
    private static volatile Nd4jBackend loadedBackend = null;

    // Thread-safe set to track loaded libraries
    private static final Set<String> loadedLibraries = ConcurrentHashMap.newKeySet();

    // Read-write lock for backend loading operations
    private static final ReentrantReadWriteLock loadLock = new ReentrantReadWriteLock();

    // Lock for library loading operations
    private static final Object libraryLoadLock = new Object();

    public static boolean IS_LOADED = false;

    // Safe initialization of CPU priority
    static {
        BACKEND_PRIORITY_CPU = initializePriority(
                ND4JSystemProperties.BACKEND_PRIORITY_CPU,
                ND4JEnvironmentVars.BACKEND_PRIORITY_CPU,
                "CPU"
        );
    }

    // Safe initialization of GPU priority
    static {
        BACKEND_PRIORITY_GPU = initializePriority(
                ND4JSystemProperties.BACKEND_PRIORITY_GPU,
                ND4JEnvironmentVars.BACKEND_PRIORITY_GPU,
                "GPU"
        );
    }

    /**
     * Safely initialize priority values with proper error handling
     */
    private static int initializePriority(String systemProperty, String envVar, String type) {
        int priority = 0;

        try {
            String value = System.getProperty(systemProperty);
            if (value != null && !value.trim().isEmpty()) {
                priority = Integer.parseInt(value.trim());
            } else {
                value = System.getenv(envVar);
                if (value != null && !value.trim().isEmpty()) {
                    priority = Integer.parseInt(value.trim());
                }
            }
        } catch (NumberFormatException e) {
            log.warn("Invalid {} backend priority value, using default 0", type, e);
            priority = 0;
        } catch (SecurityException e) {
            log.warn("Security exception accessing {} backend priority, using default 0", type, e);
            priority = 0;
        }

        return priority;
    }

    /**
     * Returns true if the backend allows order to be specified on blas operations (cblas)
     * @return true if the backend allows order to be specified on blas operations
     */
    public abstract boolean allowsOrder();

    /**
     * Gets a priority number for the backend.
     * Backends are loaded in priority order (highest first).
     * @return a priority number.
     */
    public abstract int getPriority();

    /**
     * Determines whether a given backend is available in the current environment.
     * @return true if the backend is available; false otherwise.
     */
    public abstract boolean isAvailable();

    /**
     * Returns true if the backend can run on the os or not
     * @return
     */
    public abstract boolean canRun();

    /**
     * Get the configuration resource
     * @return
     */
    public abstract Resource getConfigurationResource();

    /**
     * Get the actual (concrete/implementation) class for standard INDArrays for this backend
     */
    public abstract Class getNDArrayClass();

    public abstract Environment getEnvironment();

    /**
     * Get the build information of the backend
     */
    public abstract String buildInfo();

    /**
     * Loads the best available backend with thread safety and deadlock prevention.
     * @return the loaded backend
     * @throws NoAvailableBackendException if no backend is available
     */
    public static Nd4jBackend load() throws NoAvailableBackendException {
        // First, try to return already loaded backend (fast path)
        if (isLoaded.get() && loadedBackend != null) {
            return loadedBackend;
        }

        // Use write lock for loading operations
        loadLock.writeLock().lock();
        try {
            // Double-check after acquiring lock
            if (isLoaded.get() && loadedBackend != null) {
                return loadedBackend;
            }

            return loadBackendInternal();
        } finally {
            loadLock.writeLock().unlock();
        }
    }

    /**
     * Internal method to load the backend - must be called within write lock
     */
    private static Nd4jBackend loadBackendInternal() throws NoAvailableBackendException {
        String logInitProperty = System.getProperty(ND4JSystemProperties.LOG_INITIALIZATION, "true");
        boolean logInit = Boolean.parseBoolean(logInitProperty);

        List<Nd4jBackend> backends = new ArrayList<>();
        ServiceLoader<Nd4jBackend> loader = ND4JClassLoading.loadService(Nd4jBackend.class);

        try {
            for (Nd4jBackend nd4jBackend : loader) {
                backends.add(nd4jBackend);
            }
        } catch (ServiceConfigurationError serviceError) {
            log.warn("Failed to process available backends", serviceError);
            // Don't return null, try dynamic loading instead
        }

        // Sort backends by priority (high-priority first)
        backends.sort((o1, o2) -> Integer.compare(o2.getPriority(), o1.getPriority()));

        // Try to load available backends
        for (Nd4jBackend backend : backends) {
            if (tryLoadBackend(backend, logInit)) {
                loadedBackend = backend;
                isLoaded.set(true);
                IS_LOADED = true;
                return backend;
            }
        }

        // If no backend found, try dynamic loading
        if (tryDynamicLoading()) {
            // Recursively try loading again after dynamic loading
            return loadBackendInternal();
        }

        throw new NoAvailableBackendException(
                "Please ensure that you have an nd4j backend on your classpath. Please see: https://deeplearning4j.konduit.ai/nd4j/backend");
    }

    /**
     * Try to load a specific backend safely
     */
    private static boolean tryLoadBackend(Nd4jBackend backend, boolean logInit) {
        boolean available = false;
        String error = null;

        try {
            available = backend.isAvailable();
        } catch (Exception e) {
            error = e.getMessage();
        }

        if (!available) {
            if (logInit) {
                log.warn("Skipped [{}] backend (unavailable): {}",
                        backend.getClass().getSimpleName(), error);
            }
            return false;
        }

        try {
            Nd4jContext.getInstance().updateProperties(backend.getConfigurationResource().getInputStream());
        } catch (IOException e) {
            log.error("Failed to load configuration for backend: {}", backend.getClass().getSimpleName(), e);
            return false;
        }

        if (logInit) {
            log.info("Loaded [{}] backend with logging {}",
                    backend.getClass().getSimpleName(), log.getClass().getName());
        }

        return true;
    }

    /**
     * Try dynamic loading of backends
     */
    private static boolean tryDynamicLoading() throws NoAvailableBackendException {
        // Use atomic boolean to prevent multiple attempts
        if (!triedDynamicLoad.compareAndSet(false, true)) {
            return false; // Already tried
        }

        String[] jarUris = getDynamicLoadPaths();
        if (jarUris == null || jarUris.length == 0) {
            return false;
        }

        // Load all the discoverable URIs
        for (String uri : jarUris) {
            if (uri != null && !uri.trim().isEmpty()) {
                loadLibrary(new File(uri.trim()));
            }
        }

        return true;
    }

    /**
     * Get dynamic load paths from system properties or environment variables
     */
    private static String[] getDynamicLoadPaths() {
        try {
            String paths = System.getProperty(ND4JSystemProperties.DYNAMIC_LOAD_CLASSPATH_PROPERTY);
            if (paths != null && !paths.trim().isEmpty()) {
                return paths.split(";");
            }

            paths = System.getenv(ND4JEnvironmentVars.BACKEND_DYNAMIC_LOAD_CLASSPATH);
            if (paths != null && !paths.trim().isEmpty()) {
                return paths.split(";");
            }
        } catch (SecurityException e) {
            log.warn("Security exception accessing dynamic load paths", e);
        }

        return null;
    }

    /**
     * Adds the supplied Java Archive library to java.class.path. This is benign
     * if the library is already loaded.
     * @param jar the jar file to add
     * @throws NoAvailableBackendException
     */
    public static void loadLibrary(File jar) throws NoAvailableBackendException {
        if (jar == null || !jar.exists()) {
            log.warn("Jar file does not exist: {}", jar);
            return;
        }

        String jarPath = jar.getAbsolutePath();

        // Check if already loaded (fast path)
        if (loadedLibraries.contains(jarPath)) {
            return;
        }

        // Synchronize library loading
        synchronized (libraryLoadLock) {
            // Double-check after acquiring lock
            if (loadedLibraries.contains(jarPath)) {
                return;
            }

            try {
                URLClassLoader loader = (URLClassLoader) ND4JClassLoading.getNd4jClassloader();
                java.net.URL url = jar.toURI().toURL();

                // Check if URL is already in classpath
                for (java.net.URL existingUrl : loader.getURLs()) {
                    if (existingUrl.equals(url)) {
                        loadedLibraries.add(jarPath);
                        return;
                    }
                }

                // Add URL to classpath
                java.lang.reflect.Method method = URLClassLoader.class.getDeclaredMethod("addURL", java.net.URL.class);
                method.setAccessible(true);
                method.invoke(loader, url);

                loadedLibraries.add(jarPath);
                log.debug("Successfully loaded library: {}", jarPath);

            } catch (final NoSuchMethodException | IllegalAccessException |
                           java.net.MalformedURLException | java.lang.reflect.InvocationTargetException e) {
                throw new NoAvailableBackendException("Failed to load library: " + jarPath, e);
            }
        }
    }

    /**
     * Get backend properties thread-safely
     * @return Properties object
     * @throws IOException
     */
    public Properties getProperties() throws IOException {
        return getContext().getConf();
    }

    /**
     * Get backend context thread-safely
     * @return Nd4jContext instance
     * @throws IOException
     */
    public Nd4jContext getContext() throws IOException {
        return Nd4jContext.getInstance();
    }

    @Override
    public String toString() {
        return getClass().getName();
    }

    public abstract void logBackendInit();

    /**
     * Check if a backend is currently loaded
     * @return true if a backend is loaded
     */
    public static boolean isBackendLoaded() {
        loadLock.readLock().lock();
        try {
            return isLoaded.get() && loadedBackend != null;
        } finally {
            loadLock.readLock().unlock();
        }
    }

    /**
     * Get the currently loaded backend (if any)
     * @return the loaded backend or null if none is loaded
     */
    public static Nd4jBackend getLoadedBackend() {
        loadLock.readLock().lock();
        try {
            return isLoaded.get() ? loadedBackend : null;
        } finally {
            loadLock.readLock().unlock();
        }
    }

    @SuppressWarnings("serial")
    public static class NoAvailableBackendException extends Exception {
        public NoAvailableBackendException(String s) {
            super(s);
        }

        /**
         * Constructs a new exception with the specified cause and a detail
         * message of <tt>(cause==null ? null : cause.toString())</tt> (which
         * typically contains the class and detail message of <tt>cause</tt>).
         * This constructor is useful for exceptions that are little more than
         * wrappers for other throwables (for example, {@link
         * PrivilegedActionException}).
         *
         * @param cause the cause (which is saved for later retrieval by the
         *              {@link #getCause()} method).  (A <tt>null</tt> value is
         *              permitted, and indicates that the cause is nonexistent or
         *              unknown.)
         * @since 1.4
         */
        public NoAvailableBackendException(Throwable cause) {
            super(cause);
        }

        public NoAvailableBackendException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}