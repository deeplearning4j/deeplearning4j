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
package org.nd4j.linalg.platform.cuda;

/**
 * Build identity of the CUDA platform aggregator. The CUDA version is part of
 * the published Maven coordinates ({@code nd4j-cuda-<version>-platform}) so
 * consumers can introspect which CUDA ABI a given JAR was assembled against.
 */
public final class CudaPlatform {
    /** CUDA ABI this platform aggregator was assembled against. */
    public static final String CUDA_VERSION = "12.9";

    private CudaPlatform() {
    }
}
