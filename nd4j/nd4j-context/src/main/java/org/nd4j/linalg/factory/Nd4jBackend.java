/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.linalg.factory;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.config.ND4JEnvironmentVars;
import org.nd4j.config.ND4JSystemProperties;
import org.nd4j.context.Nd4jContext;
import org.nd4j.linalg.io.Resource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.security.PrivilegedActionException;
import java.util.*;

/**
 * An ND4j backend.
 *
 * A "backend" is also described here: http://nd4j.org/backend.html
 *
 * A backend also has 2  variables to be aware of.
 * 1 is the environment variable, ND4J_DYNAMIC_LOAD_CLASSPATH
 * This will define a uri path separated by ; where jars will be
 * loaded from the path and dynamically loaded.
 *
 * The other is the system property:
 * org.nd4j.backend.dynamicbackend
 *
 * This has the same use case but is for system properties.
 * Of note here is that the system property takes loading precedence over
 * the environment variable. If you want to just use the environment variable,
 * don't define the system property.
 *
 * Both of these variables are for dynamically loading a backend relative to a path.
 * The main idea here is for distributed environments like spark where
 * you have multiple worker nodes with some having gpus and others not.
 *
 * When you define an environment variable on the server, you can
 * have a hardware jar file load with respect to the node nd4j is installed on.
 * The system property is mainly for flexibility and probably shouldn't be
 * used in practice.
 *
 * @author eronwright
 * @author Adam Gibson
 * @author saudet
 *
 */
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
    private static boolean triedDynamicLoad = false;

    static {
        int n = 0;
        String s = System.getenv(ND4JEnvironmentVars.BACKEND_PRIORITY_CPU);
        if (s != null && s.length() > 0) {
            try {
                n = Integer.parseInt(s);
            } catch (NumberFormatException e) {
                throw new RuntimeException(e);
            }
        }
        BACKEND_PRIORITY_CPU = n;
    }

    static {
        int n = 100;
        String s = System.getenv(ND4JEnvironmentVars.BACKEND_PRIORITY_GPU);
        if (s != null && s.length() > 0) {
            try {
                n = Integer.parseInt(s);
            } catch (NumberFormatException e) {
                throw new RuntimeException(e);
            }
        }
        BACKEND_PRIORITY_GPU = n;
    }

    /**
     * Returns true if the
     * backend allows order to be specified
     * on blas operations (cblas)
     * @return true if the backend allows
     * order to be specified on blas operations
     */
    public abstract boolean allowsOrder();

    /**
     * Gets a priority number for the backend.
     *
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
     * Returns true if the backend can
     * run on the os or not
     * @return
     */
    public abstract boolean canRun();

    /**
     * Get the configuration resource
     * @return
     */
    public abstract Resource getConfigurationResource();

    /**
     *  Get the actual (concrete/implementation) class for standard INDArrays for this backend
     */
    public abstract Class getNDArrayClass();


    /**
     * Loads the best available backend.
     * @return
     */
    public static Nd4jBackend load() throws NoAvailableBackendException {

        List<Nd4jBackend> backends = new ArrayList<>(1);
        ServiceLoader<Nd4jBackend> loader = ServiceLoader.load(Nd4jBackend.class);
        try {

            Iterator<Nd4jBackend> backendIterator = loader.iterator();
            while (backendIterator.hasNext())
                backends.add(backendIterator.next());

        } catch (ServiceConfigurationError serviceError) {
            // a fatal error due to a syntax or provider construction error.
            // backends mustn't throw an exception during construction.
            throw new RuntimeException("failed to process available backends", serviceError);
        }

        Collections.sort(backends, new Comparator<Nd4jBackend>() {
            @Override
            public int compare(Nd4jBackend o1, Nd4jBackend o2) {
                // high-priority first
                return o2.getPriority() - o1.getPriority();
            }
        });

        for (Nd4jBackend backend : backends) {
            boolean available = false;
            String error = null;
            try {
                available = backend.isAvailable();
            } catch (Exception e) {
                error = e.getMessage();
            }
            if (!available) {
                log.warn("Skipped [{}] backend (unavailable): {}", backend.getClass().getSimpleName(), error);
                continue;
            }

            try {
                Nd4jContext.getInstance().updateProperties(backend.getConfigurationResource().getInputStream());
            } catch (IOException e) {
                e.printStackTrace();
            }

            log.info("Loaded [{}] backend", backend.getClass().getSimpleName());
            return backend;
        }

        //need to dynamically load jars and recall, note that we do this right before the backend loads.
        //An existing backend should take precedence over
        //ones being dynamically discovered.
        //Note that we prioritize jvm properties first, followed by environment variables.
        String[] jarUris;
        if (System.getProperties().containsKey(ND4JSystemProperties.DYNAMIC_LOAD_CLASSPATH_PROPERTY) && !triedDynamicLoad) {
            jarUris = System.getProperties().getProperty(ND4JSystemProperties.DYNAMIC_LOAD_CLASSPATH_PROPERTY).split(";");
        // Do not call System.getenv(): Accessing all variables requires higher security privileges
        } else if (System.getenv(ND4JEnvironmentVars.BACKEND_DYNAMIC_LOAD_CLASSPATH) != null && !triedDynamicLoad) {
            jarUris = System.getenv(ND4JEnvironmentVars.BACKEND_DYNAMIC_LOAD_CLASSPATH).split(";");
        }

        else
            throw new NoAvailableBackendException(
                            "Please ensure that you have an nd4j backend on your classpath. Please see: http://nd4j.org/getstarted.html");

        triedDynamicLoad = true;
        //load all the discoverable uris and try to load the backend again
        for (String uri : jarUris) {
            loadLibrary(new File(uri));
        }

        return load();
    }


    /**
     * Adds the supplied Java Archive library to java.class.path. This is benign
     * if the library is already loaded.
     * @param jar the jar file to add
     * @throws NoAvailableBackendException
     */
    public static synchronized void loadLibrary(File jar) throws NoAvailableBackendException {
        try {
            /*We are using reflection here to circumvent encapsulation; addURL is not public*/
            java.net.URLClassLoader loader = (java.net.URLClassLoader) ClassLoader.getSystemClassLoader();
            java.net.URL url = jar.toURI().toURL();
            /*Disallow if already loaded*/
            for (java.net.URL it : java.util.Arrays.asList(loader.getURLs())) {
                if (it.equals(url)) {
                    return;
                }
            }
            java.lang.reflect.Method method =
                            java.net.URLClassLoader.class.getDeclaredMethod("addURL", new Class[] {java.net.URL.class});
            method.setAccessible(true); /*promote the method to public access*/
            method.invoke(loader, new Object[] {url});
        } catch (final java.lang.NoSuchMethodException | java.lang.IllegalAccessException
                        | java.net.MalformedURLException | java.lang.reflect.InvocationTargetException e) {
            throw new NoAvailableBackendException(e);
        }
    }

    /**
     *
     * @return
     * @throws IOException
     */
    public Properties getProperties() throws IOException {
        return getContext().getConf();
    }

    /**
     *
     * @return
     * @throws IOException
     */
    public Nd4jContext getContext() throws IOException {
        return Nd4jContext.getInstance();
    }

    @Override
    public String toString() {
        return getClass().getName();
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
    }
}
