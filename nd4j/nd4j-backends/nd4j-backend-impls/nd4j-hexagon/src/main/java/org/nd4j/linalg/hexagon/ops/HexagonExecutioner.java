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

package org.nd4j.linalg.hexagon.ops;

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
import org.nd4j.linalg.hexagon.HexagonEnvironment;

import java.util.Properties;

/**
 * Hexagon NPU Operation Executioner skeleton (hexagon-mlir).
 *
 * All execution entry points throw UnsupportedOperationException until the hexagon-mlir native
 * bindings exist (see ADR 0088). {@link org.nd4j.linalg.hexagon.HexagonBackend#canRun()} returns
 * false, so this executioner can never be selected at runtime — the skeleton exists so the module
 * tracks the current OpExecutioner contract and fails to compile (rather than silently rotting)
 * when that contract changes. HexagonBackendSmokeTest locks the canRun() contract.
 *
 * The intended implementation compiles ops/graphs to MLIR targeting HVX vector operations via
 * hexagon-mlir (BSD-3, open-sourced by Qualcomm Dec 2025), stages data through TCM, and replays
 * recorded command lists for minimal dispatch overhead — INT8-first.
 */
@Slf4j
public class HexagonExecutioner extends DefaultOpExecutioner {

    private static final String NOT_IMPLEMENTED =
            "Hexagon NPU execution requires hexagon-mlir native bindings (not yet implemented) — see ADR 0088";

    public HexagonExecutioner() {
        super();
    }

    public String getExecutionerType() {
        return "HEXAGON";
    }

    @Override
    public INDArray exec(Op op) {
        throw new UnsupportedOperationException(NOT_IMPLEMENTED);
    }

    @Override
    public INDArray exec(Op op, OpContext opContext) {
        throw new UnsupportedOperationException(NOT_IMPLEMENTED);
    }

    @Override
    public INDArray exec(org.nd4j.linalg.api.ops.ScalarOp op) {
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
    public ExecutionerType type() {
        return ExecutionerType.HEXAGON;
    }

    @Override
    public boolean isExperimentalMode() {
        return false;
    }

    @Override
    public Properties getEnvironmentInformation() {
        HexagonEnvironment env = HexagonEnvironment.getInstance();
        Properties props = new Properties();
        props.setProperty("backend", "HEXAGON");
        props.setProperty("runtime", "hexagon-mlir");
        props.setProperty("int8.support", "true");
        props.setProperty("hexagon.npu.version", env.getNpuVersion());
        props.setProperty("hexagon.hvx.width.bytes", String.valueOf(env.getHvxVectorWidth()));
        props.setProperty("hexagon.tcm.bytes", String.valueOf(env.getTcmCapacity()));
        return props;
    }
}
