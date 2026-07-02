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

package org.nd4j.linalg.jtpu;

import lombok.extern.slf4j.Slf4j;

/**
 * Placeholder for the TPU NDArray factory.
 *
 * NOTE: this class intentionally does NOT extend
 * {@link org.nd4j.linalg.factory.BaseNDArrayFactory} yet. The original scaffolding targeted an
 * old revision of the factory API and bit-rotted into an unbuildable state (see ADR 0102). A real
 * factory implementation only makes sense once the PJRT native bindings exist — PJRT arrays are
 * device buffers created through a PJRT client, which shapes the factory design (see ADR 0072).
 *
 * {@link JTpuBackend#canRun()} returns false until those bindings land, so no code path can
 * request this factory. TpuBackendSmokeTest locks that contract.
 */
@Slf4j
public class JTpuNDArrayFactory {

    public JTpuNDArrayFactory() {
        log.debug("JTpuNDArrayFactory placeholder constructed — PJRT bindings not yet implemented");
    }
}
