/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  * *****************************************************************************
 */

package org.nd4j.nativeimage;

import org.graalvm.nativeimage.hosted.Feature;
import org.graalvm.nativeimage.hosted.RuntimeClassInitialization;
import org.graalvm.nativeimage.hosted.RuntimeJNIAccess;
import org.graalvm.nativeimage.hosted.RuntimeReflection;
import org.nd4j.autodiff.functions.DifferentialFunction;

import java.util.Set;

/**
 * Owns ND4J's dynamic Native Image reachability contract.
 *
 * <p>{@code DifferentialFunctionClassHolder} creates operation prototypes
 * reflectively at runtime. JavaCPP also reaches generated backend bindings,
 * including their nested pointer and callback types, through both reflection
 * and JNI. Both catalogs are derived from the effective application classpath
 * here instead of requiring each backend or application to maintain a partial
 * hand-written list.</p>
 */
public final class Nd4jOpsReflectionFeature implements Feature {

    @Override
    public void beforeAnalysis(BeforeAnalysisAccess access) {
        Set<Class<? extends DifferentialFunction>> operations =
                Nd4jOperationClassScanner.discover();

        for (Class<? extends DifferentialFunction> operation : operations) {
            registerReflection(operation);
        }

        Set<Class<?>> javaCppBindings =
                Nd4jJavaCppClassScanner.discoverBindingClasses(
                        access.getApplicationClassPath(),
                        access.getApplicationClassLoader());
        for (Class<?> binding : javaCppBindings) {
            registerReflection(binding);
            registerJni(binding);
            RuntimeClassInitialization.initializeAtRunTime(binding);
        }

        System.err.println("[native-image] ND4J registered "
                + operations.size()
                + " operation classes for runtime reflection and "
                + javaCppBindings.size()
                + " JavaCPP binding classes for runtime reflection and JNI");
    }

    private static void registerReflection(Class<?> type) {
        RuntimeReflection.register(type);
        RuntimeReflection.register(type.getDeclaredConstructors());
        RuntimeReflection.register(type.getDeclaredMethods());
        RuntimeReflection.register(type.getDeclaredFields());
    }

    private static void registerJni(Class<?> type) {
        RuntimeJNIAccess.register(type);
        RuntimeJNIAccess.register(type.getDeclaredConstructors());
        RuntimeJNIAccess.register(type.getDeclaredMethods());
        RuntimeJNIAccess.register(type.getDeclaredFields());
    }
}
