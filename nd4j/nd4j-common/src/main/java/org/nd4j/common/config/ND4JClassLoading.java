/*******************************************************************************
 * Copyright (c) Eclipse Deeplearning4j Contributors 2020
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

package org.nd4j.common.config;

import lombok.extern.slf4j.Slf4j;

import java.util.ServiceLoader;

/**
 * Global context for class-loading in ND4J.
 * <p>Use {@code ND4JClassLoading} to define classloader for ND4J only! To define classloader used by
 * {@code Deeplearning4j} use class {@link org.deeplearning4j.common.config.DL4JClassLoading}.
 *
 * <p>Usage:
 * <pre>{@code
 * public class Application {
 *     static {
 *         ND4JClassLoading.setNd4jClassloaderFromClass(Application.class);
 *     }
 *
 *     public static void main(String[] args) {
 *     }
 * }
 * }</code>
 *
 * @see org.deeplearning4j.common.config.DL4JClassLoading
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
