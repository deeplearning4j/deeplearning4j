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

package org.nd4j.linalg.jtpu.ops;

import lombok.extern.slf4j.Slf4j;

/**
 * Placeholder for the TPU op context.
 *
 * NOTE: this class intentionally does NOT implement {@link org.nd4j.linalg.api.ops.OpContext}
 * yet. The original scaffolding targeted an old revision of that interface and bit-rotted into an
 * unbuildable state (see ADR 0102). A real op context only makes sense once the PJRT native
 * bindings exist — a PJRT op context wraps device buffers and a PJRT_LoadedExecutable, which
 * shapes the design (see ADR 0072).
 *
 * {@link TpuExecutioner#buildContext()} throws UnsupportedOperationException until then, and
 * {@link org.nd4j.linalg.jtpu.JTpuBackend#canRun()} returns false so no code path can reach it.
 * TpuBackendSmokeTest locks that contract.
 */
@Slf4j
public class TpuOpContext {

    public TpuOpContext() {
        log.debug("TpuOpContext placeholder constructed — PJRT bindings not yet implemented");
    }
}
