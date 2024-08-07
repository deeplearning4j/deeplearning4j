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
package org.nd4j.common.tools;

import org.nd4j.common.config.ND4JClassLoading;

/**
 * Utility which ensures that classes are loaded by the {@link ClassLoader}.
 * //Pulled from Netty here under the apache license v 2.0:
 * https://github.com/netty/netty/blob/38086002024ad274aa0f9c168b4e47555a423836/common/src/main/java/io/netty/util/internal/ClassInitializerUtil.java
 */
public final class ClassInitializerUtil {

    private ClassInitializerUtil() { }

    /**
     * Preload the given classes and so ensure the {@link ClassLoader} has these loaded after this method call.
     *
     * @param loadingClass      the {@link Class} that wants to load the classes.
     * @param classes           the classes to load.
     */
    public static void tryLoadClasses(Class<?> loadingClass, Class<?>... classes) {
        ClassLoader loader = ND4JClassLoading.getNd4jClassloader();
        for (Class<?> clazz: classes) {
            tryLoadClass(loader, clazz.getName());
        }
    }

    private static void tryLoadClass(ClassLoader classLoader, String className) {
        try {
            // Load the class and also ensure we init it which means its linked etc.
            Class.forName(className, true, classLoader);
        } catch (ClassNotFoundException ignore) {
            // Ignore
        } catch (SecurityException ignore) {
            // Ignore
        }
    }
}