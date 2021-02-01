/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.listeners.debugging;

import lombok.*;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.util.ArrayUtil;

import java.text.DecimalFormat;
import java.util.*;

/**
 * A simple listener for benchmarking single operations in SameDiff<br>
 * Supports 2 modes:<br>
 * - SINGLE_ITER_PRINT: Print the runtime of the first iteration<br>
 * - AGGREGATE: Collect statistics for multiple runs, that can be accessed (by op name) via {@link #getAggregateModeMap()}
 *
 * @author Alex Black
 */
@Getter
public class OpBenchmarkListener extends BaseListener {

    public enum Mode {SINGLE_ITER_PRINT, AGGREGATE}

    private final Operation operation;
    private final Mode mode;
    private final long minRuntime;
    private Map<String,OpExec> aggregateModeMap;

    @Getter(AccessLevel.PRIVATE)
    private long start;
    @Getter(AccessLevel.PRIVATE)
    private boolean printActive;
    private boolean printDone;

    public OpBenchmarkListener(Operation operation, @NonNull Mode mode) {
        this(operation, mode, 0);
    }

    /**
     * @param operation  Operation to collect stats for
     * @param mode       Mode - see {@link OpBenchmarkListener}
     * @param minRuntime Minimum runtime - only applies to Mode.SINGLE_ITER_PRINT. If op runtime below this: don't print
     */
    public OpBenchmarkListener(Operation operation, @NonNull Mode mode, long minRuntime) {
        this.operation = operation;
        this.mode = mode;
        this.minRuntime = minRuntime;
    }

    @Override
    public boolean isActive(Operation operation) {
        return this.operation == null || this.operation == operation;
    }

    @Override
    public void operationStart(SameDiff sd, Operation op) {
        if(printDone)
            return;
        if(this.operation == null || this.operation == op)
            printActive = true;
    }

    @Override
    public void operationEnd(SameDiff sd, Operation op) {
        if(printDone)
            return;
        if(this.operation == null || this.operation == op) {
            printActive = false;
            printDone = true;
        }
    }

    @Override
    public void preOpExecution(SameDiff sd, At at, SameDiffOp op, OpContext opContext) {
        start = System.currentTimeMillis();
    }

    @Override
    public void opExecution(SameDiff sd, At at, MultiDataSet batch, SameDiffOp op, OpContext opContext, INDArray[] outputs) {
        long now = System.currentTimeMillis();

        if (mode == Mode.SINGLE_ITER_PRINT && printActive && (now-start) > this.minRuntime) {
            System.out.println(getOpString(op, now));
        } else if (mode == Mode.AGGREGATE) {
            if(aggregateModeMap == null)
                aggregateModeMap = new LinkedHashMap<>();

            if(!aggregateModeMap.containsKey(op.getName())){
                String s = getOpString(op, null);
                OpExec oe = new OpExec(op.getName(), op.getOp().opName(), op.getOp().getClass(),
                        new ArrayList<Long>(), s);
                aggregateModeMap.put(op.getName(), oe);
            }

            aggregateModeMap.get(op.getName()).getRuntimeMs().add(now-start);
        }
    }

    private String getOpString(SameDiffOp op, Long now){
        StringBuilder sb = new StringBuilder();
        sb.append(op.getName()).append(" - ").append(op.getOp().getClass().getSimpleName())
                .append("(").append(op.getOp().opName()).append(") - ");
        if(now != null) {
            sb.append(now - start).append(" ms\n");
        }

        if (op.getOp() instanceof DynamicCustomOp) {
            DynamicCustomOp dco = (DynamicCustomOp) op.getOp();
            int x = 0;

            for (INDArray i : dco.inputArguments()) {
                sb.append("  in ").append(x++).append(": ").append(i.shapeInfoToString()).append("\n");
            }
            x = 0;
            for (INDArray o : dco.outputArguments()) {
                sb.append("  out ").append(x++).append(": ").append(o.shapeInfoToString()).append("\n");
            }
            long[] iargs = dco.iArgs();
            boolean[] bargs = dco.bArgs();
            double[] targs = dco.tArgs();
            if (iargs != null && iargs.length > 0) {
                sb.append("  iargs: ").append(Arrays.toString(iargs)).append("\n");
            }
            if (bargs != null && bargs.length > 0) {
                sb.append("  bargs: ").append(Arrays.toString(bargs)).append("\n");
            }
            if (targs != null && targs.length > 0) {
                sb.append("  targs: ").append(Arrays.toString(targs)).append("\n");
            }
        } else {
            Op o = (Op) op.getOp();
            if (o.x() != null)
                sb.append("  x: ").append(o.x().shapeInfoToString());
            if (o.y() != null)
                sb.append("  y: ").append(o.y().shapeInfoToString());
            if (o.z() != null)
                sb.append("  z: ").append(o.z().shapeInfoToString());
        }
        return sb.toString();
    }


    @AllArgsConstructor
    @Data
    public static class OpExec {
        private final String opOwnName;
        private final String opName;
        private final Class<?> opClass;
        private List<Long> runtimeMs;
        private String firstIter;

        @Override
        public String toString(){
            DecimalFormat df = new DecimalFormat("0.000");

            return opOwnName + " - op class: " + opClass.getSimpleName() + " (op name: " + opName + ")\n"
                    + "count: " + runtimeMs.size() + ", mean: " + df.format(avgMs()) + "ms, std: " + df.format(stdMs()) + "ms, min: " + minMs() + "ms, max: " + maxMs() + "ms\n"
                    + firstIter;
        }

        public double avgMs() {
            long sum = 0;
            for (Long l : runtimeMs) {
                sum += l;
            }
            return sum / (double) runtimeMs.size();
        }

        public double stdMs() {
            return Nd4j.createFromArray(ArrayUtil.toArrayLong(runtimeMs)).stdNumber().doubleValue();
        }

        public long minMs() {
            return Nd4j.createFromArray(ArrayUtil.toArrayLong(runtimeMs)).minNumber().longValue();
        }

        public long maxMs() {
            return Nd4j.createFromArray(ArrayUtil.toArrayLong(runtimeMs)).maxNumber().longValue();
        }
    }
}
