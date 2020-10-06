package org.nd4j.common.config;

import lombok.extern.slf4j.Slf4j;

import java.util.ServiceLoader;

/**
 * Global context for class-loading in ND4J.
 *
 * @author Alexei KLENIN
 */
@Slf4j
public final class ND4JClassLoading {
    private static ClassLoader nd4jClassloader = Thread.currentThread().getContextClassLoader();

    private ND4JClassLoading() {
    }

    public static ClassLoader getNd4jClassloader() {
        return ND4JClassLoading.nd4jClassloader;
    }

    public static void setNd4jClassloaderFromClass(Class<?> clazz) {
        setNd4jClassloader(clazz.getClassLoader());
    }

    public static void setNd4jClassloader(ClassLoader nd4jClassloader) {
        ND4JClassLoading.nd4jClassloader = nd4jClassloader;
        log.debug("Global class-loader for ND4J was changed.");
    }

    public static boolean classPresentOnClasspath(String className) {
        return classPresentOnClasspath(className, nd4jClassloader);
    }

    public static boolean classPresentOnClasspath(String className, ClassLoader classLoader) {
        return loadClassByName(className, false, classLoader) != null;
    }

    public static <T> Class<T> loadClassByName(String className) {
        return loadClassByName(className, true, nd4jClassloader);
    }

    @SuppressWarnings("unchecked")
    public static <T> Class<T> loadClassByName(String className, boolean initialize, ClassLoader classLoader) {
        try {
            return (Class<T>) Class.forName(className, initialize, classLoader);
        } catch (ClassNotFoundException classNotFoundException) {
            log.error(String.format("Cannot find class [%s] of provided class-loader.", className));
            return null;
        }
    }

    public static <S> ServiceLoader<S> loadService(Class<S> serviceClass) {
        return loadService(serviceClass, nd4jClassloader);
    }

    public static <S> ServiceLoader<S> loadService(Class<S> serviceClass, ClassLoader classLoader) {
        return ServiceLoader.load(serviceClass, classLoader);
    }
}
