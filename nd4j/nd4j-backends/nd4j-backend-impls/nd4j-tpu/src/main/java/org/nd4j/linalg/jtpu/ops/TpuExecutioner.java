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
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BroadcastOp;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.RandomOp;
import org.nd4j.linalg.api.ops.ReduceOp;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.api.rng.Random;

import java.util.Properties;

/**
 * TPU Operation Executioner skeleton (PJRT).
 *
 * All execution entry points throw UnsupportedOperationException until the PJRT native bindings
 * exist (see ADR 0072). {@link org.nd4j.linalg.jtpu.JTpuBackend#canRun()} returns false, so this
 * executioner can never be selected at runtime — the skeleton exists so the module tracks the
 * current OpExecutioner contract and fails to compile (rather than silently rotting) when that
 * contract changes. TpuBackendSmokeTest locks the canRun() contract.
 *
 * The intended implementation compiles ops/graphs to HLO and executes them through the PJRT C API
 * (libtpu), with compiled-executable caching and bfloat16-native execution.
 */
@Slf4j
public class TpuExecutioner extends DefaultOpExecutioner {

    private static final String NOT_IMPLEMENTED =
            "TPU execution requires PJRT native bindings (not yet implemented) — see ADR 0072";

    public TpuExecutioner() {
        super();
    }

    public String getExecutionerType() {
        return "TPU";
    }

    @Override
    public INDArray exec(Op op) {
        throw new UnsupportedOperationException(NOT_IMPLEMENTED);
    }

    @Override
    public INDArray exec(org.nd4j.linalg.api.ops.ScalarOp op) {
        throw new UnsupportedOperationException(NOT_IMPLEMENTED);
    }

    @Override
    public org.nd4j.linalg.cache.TADManager getTADManager() {
        throw new UnsupportedOperationException(NOT_IMPLEMENTED);
    }

    @Override
    public void registerGraph(long id, org.bytedeco.javacpp.Pointer graph) {
        throw new UnsupportedOperationException(NOT_IMPLEMENTED);
    }

    @Override
    public void forgetGraph(long id) {
        throw new UnsupportedOperationException(NOT_IMPLEMENTED);
    }

    @Override
    public OpContext buildContext() {
        throw new UnsupportedOperationException(NOT_IMPLEMENTED);
    }

    @Override
    public INDArray exec(Op op, OpContext opContext) {
        throw new UnsupportedOperationException(NOT_IMPLEMENTED);
    }

    @Override
    public INDArray exec(ReduceOp op) {
        throw new UnsupportedOperationException(NOT_IMPLEMENTED);
    }

    @Override
    public INDArray exec(Variance accumulation) {
        throw new UnsupportedOperationException(NOT_IMPLEMENTED);
    }

    @Override
    public INDArray exec(IndexAccumulation op) {
        throw new UnsupportedOperationException(NOT_IMPLEMENTED);
    }

    @Override
    public INDArray exec(BroadcastOp broadcast) {
        throw new UnsupportedOperationException(NOT_IMPLEMENTED);
    }

    @Override
    public INDArray exec(RandomOp op, Random rng) {
        throw new UnsupportedOperationException(NOT_IMPLEMENTED);
    }

    @Override
    public INDArray createFromDescriptor(DataBuffer shapeInformation) {
        throw new UnsupportedOperationException(NOT_IMPLEMENTED);
    }

    @Override
    public ExecutionerType type() {
        return ExecutionerType.TPU;
    }

    @Override
    public boolean isExperimentalMode() {
        return false;
    }

    @Override
    public Properties getEnvironmentInformation() {
        Properties props = new Properties();
        props.setProperty("backend", "TPU");
        props.setProperty("runtime", "PJRT");
        props.setProperty("bfloat16.support", "true");
        return props;
    }
}
