/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.sdx.aot;

import com.oracle.svm.core.annotate.Substitute;
import com.oracle.svm.core.annotate.TargetClass;
import org.bytedeco.javacpp.Pointer;

/**
 * Native Image safety constraints for JavaCPP calls that require a host JVM JNI environment.
 *
 * <p>An embedded native image has its own Java runtime and must not call JavaCPP helpers that
 * marshal image-heap objects through a host VM's {@code JNIEnv}. Native backends resolve optional
 * function entry points through {@code NativeOps.initializeFunctionsFromProcessSymbols()}.
 * Both the public JavaCPP wrapper and its JNI helper are replaced with a deterministic failure,
 * covering every backend and initialization path included in the image.</p>
 */

final class SdxNativeImageJavaCppSafety {

    static final String FAILURE_PREFIX =
            "Embedded native images must resolve process symbols through NativeOps, not Loader.addressof: ";

    private SdxNativeImageJavaCppSafety() {
    }

    static IllegalStateException processSymbolsRequired(String name) {
        return new IllegalStateException(FAILURE_PREFIX + name);
    }
}

@TargetClass(className = "org.bytedeco.javacpp.Loader")
final class Target_org_bytedeco_javacpp_Loader {

    @Substitute
    public static Pointer addressof(String name) {
        throw SdxNativeImageJavaCppSafety.processSymbolsRequired(name);
    }
}

@TargetClass(className = "org.bytedeco.javacpp.Loader$Helper")
final class Target_org_bytedeco_javacpp_Loader_Helper {

    @Substitute
    public static Pointer addressof(String name) {
        throw SdxNativeImageJavaCppSafety.processSymbolsRequired(name);
    }
}
