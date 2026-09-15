/*
 * /* ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  ******************************************************************************/
package org.nd4j.linalg.platform.zluda;

/**
 * Build identity of the ZLUDA platform aggregator. The concrete values are
 * supplied by the release build so consumers can introspect which CUDA ABI
 * and ROCm user-space runtime a given {@code nd4j-zluda-12.9-platform} JAR
 * was assembled against.
 */
public final class ZludaPlatform {
    /** CUDA ABI the ZLUDA build executes binaries from. */
    public static final String CUDA_VERSION = "12.9";

    /** ROCm user-space runtime bundled into the native classifier. */
    public static final String ROCM_VERSION = "7.2.4";

    private ZludaPlatform() {
    }
}
