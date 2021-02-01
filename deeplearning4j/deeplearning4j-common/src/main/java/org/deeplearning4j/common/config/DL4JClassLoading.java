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

package org.deeplearning4j.common.config;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.config.ND4JClassLoading;

import java.lang.reflect.InvocationTargetException;
import java.util.Objects;
import java.util.ServiceLoader;

/**
 * Global context for class-loading in DL4J.
 * <p>Use {@code DL4JClassLoading} to define classloader for Deeplearning4j only! To define classloader used by
 * {@code ND4J} use class {@link org.nd4j.common.config.ND4JClassLoading}.
 *
 * <p>Usage:
 * <pre>{@code
 * public class Application {
 *     static {
 *         DL4JClassLoading.setDl4jClassloaderFromClass(Application.class);
 *     }
 *
 *     public static void main(String[] args) {
 *     }
 * }
 * }</code>
 *
 * @see org.nd4j.common.config.ND4JClassLoading
 *
 * @author Alexei KLENIN
 */
@Slf4j
public class DL4JClassLoading {
    private static ClassLoader dl4jClassloader = ND4JClassLoading.getNd4jClassloader();

    private DL4JClassLoading() {
    }

    public static ClassLoader getDl4jClassloader() {
        return DL4JClassLoading.dl4jClassloader;
    }

    public static void setDl4jClassloaderFromClass(Class<?> clazz) {
        setDl4jClassloader(clazz.getClassLoader());
    }

    public static void setDl4jClassloader(ClassLoader dl4jClassloader) {
        DL4JClassLoading.dl4jClassloader = dl4jClassloader;
        log.debug("Global class-loader for DL4J was changed.");
    }

    public static boolean classPresentOnClasspath(String className) {
        return classPresentOnClasspath(className, dl4jClassloader);
    }

    public static boolean classPresentOnClasspath(String className, ClassLoader classLoader) {
        return loadClassByName(className, false, classLoader) != null;
    }

    public static <T> Class<T> loadClassByName(String className) {
        return loadClassByName(className, true, dl4jClassloader);
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

    public static <T> T createNewInstance(String className, Object... args) {
        return createNewInstance(className, Object.class, args);
    }

    public static <T> T createNewInstance(String className, Class<? super T> superclass) {
        return createNewInstance(className, superclass, new Class<?>[]{}, new Object[]{});
    }

    public static <T> T createNewInstance(String className, Class<? super T> superclass, Object... args) {
        Class<?>[] parameterTypes = new Class<?>[args.length];
        for (int i = 0; i < args.length; i++) {
            Object arg = args[i];
            Objects.requireNonNull(arg);
            parameterTypes[i] = arg.getClass();
        }

        return createNewInstance(className, superclass, parameterTypes, args);
    }

    public static <T> T createNewInstance(
            String className,
            Class<? super T> superclass,
            Class<?>[] parameterTypes,
            Object... args) {
        try {
            return (T) DL4JClassLoading
                    .loadClassByName(className)
                    .asSubclass(superclass)
                    .getDeclaredConstructor(parameterTypes)
                    .newInstance(args);
        } catch (InstantiationException | IllegalAccessException | InvocationTargetException
                | NoSuchMethodException instantiationException) {
            log.error(String.format("Cannot create instance of class '%s'.", className), instantiationException);
            throw new RuntimeException(instantiationException);
        }
    }

    public static <S> ServiceLoader<S> loadService(Class<S> serviceClass) {
        return loadService(serviceClass, dl4jClassloader);
    }

    public static <S> ServiceLoader<S> loadService(Class<S> serviceClass, ClassLoader classLoader) {
        return ServiceLoader.load(serviceClass, classLoader);
    }
}
