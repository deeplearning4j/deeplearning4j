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

package org.nd4j.autodiff.listeners.debugging;

import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.BaseListener;
import org.nd4j.autodiff.listeners.Operation;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.ScalarOp;

import java.util.Arrays;

public class ExecDebuggingListener extends BaseListener {

    public enum PrintMode {OPS_ONLY, SHAPES_ONLY, REPRODUCE}

    private final PrintMode printMode;
    private final int maxIterations;
    private final boolean logIter;

    private long printIterations = 0;
    private int lastIter = -1;
    private int stepThisIter = 0;

    /**
     * @param printMode     Print mode, see {@link PrintMode}
     * @param maxIterations Maximum number of iterations to print. <= 0 for "all iterations"
     * @param logIter       If true: prefix iteration/epoch, such as "(iter=1,epoch=0,op=3)" to the output
     */
    public ExecDebuggingListener(PrintMode printMode, int maxIterations, boolean logIter){
        this.printMode = printMode;
        this.maxIterations = maxIterations;
        this.logIter = logIter;
    }

    @Override
    public boolean isActive(Operation operation) {
        return true;
    }

    @Override
    public void preOpExecution(SameDiff sd, At at, SameDiffOp op, OpContext opContext) {
        if(lastIter != at.iteration()){
            lastIter = at.iteration();
            stepThisIter = 0;
            printIterations++;
        }

        if(maxIterations > 0 && printIterations > maxIterations){
            return;
        }

        StringBuilder sb = new StringBuilder();
        if(logIter){
            sb.append("(iter=").append(at.iteration())
                    .append(",epoch=").append(at.epoch())
                    .append(",");
        }
        sb.append("op=").append(stepThisIter++)
                .append(logIter ? ") " : " - ");

        DifferentialFunction df = op.getOp();
        sb.append(op.getOp().getClass().getName());
        CustomOp co = df instanceof CustomOp ? (CustomOp) df : null;
        Op lOp = df instanceof Op ? (Op) df : null;
        if(printMode == PrintMode.OPS_ONLY){
            sb.append("\n");
        } else if(printMode == PrintMode.SHAPES_ONLY){
            if(co != null){
                if(co.iArgs() != null && co.iArgs().length > 0) {
                    sb.append("\n\tiArgs=").append(Arrays.toString(co.iArgs()));
                }
                if(co.bArgs() != null && co.bArgs().length > 0) {
                    sb.append("\n\tbArgs=").append(Arrays.toString(co.bArgs()));
                }
                if(co.tArgs() != null && co.tArgs().length > 0) {
                    sb.append("\n\ttArgs=").append(Arrays.toString(co.tArgs()));
                }
                val inputs = co.inputArguments();
                val outputs = co.outputArguments();
                if(inputs != null ) {
                    for (int i = 0; i < inputs.size(); i++) {
                        sb.append("\n\tInput[").append(i).append("]=").append(inputs.get(i).shapeInfoToString());
                    }
                }
                if(outputs != null ) {
                    for (int i = 0; i < outputs.size(); i++) {
                        sb.append("\n\tOutputs[").append(i).append("]=").append(outputs.get(i).shapeInfoToString());
                    }
                }
            } else {
                if(lOp.x() != null) {
                    sb.append("\n\tx: ").append(lOp.x().shapeInfoToString());
                }
                if(lOp.y() != null) {
                    sb.append("\n\ty: ").append(lOp.y().shapeInfoToString());
                }
                if(lOp.z() != null) {
                    sb.append("\n\tz: ").append(lOp.z().shapeInfoToString());
                }
                if(lOp instanceof ScalarOp){
                    INDArray scalar = ((ScalarOp)lOp).scalar();
                    if(scalar != null){
                        sb.append("\n\tscalar: ").append(scalar.shapeInfoToString());
                    }
                }
            }
            sb.append("\n");
        } else if(printMode == PrintMode.REPRODUCE){
            sb.append("\n");
            if(co != null){
                sb.append("DynamicCustomOp op = new ").append(co.getClass().getName()).append("();\n");
                if(co.iArgs() != null && co.iArgs().length > 0 ){
                    sb.append("op.addIArgument(").append(Arrays.toString(co.iArgs()).replaceAll("[\\[\\]]", "")).append(");\n");
                }
                if(co.bArgs() != null && co.bArgs().length > 0 ){
                    sb.append("op.addBArgument(").append(Arrays.toString(co.bArgs()).replaceAll("[\\[\\]]", "")).append(");\n");
                }
                if(co.tArgs() != null && co.tArgs().length > 0 ){
                    sb.append("op.addTArgument(").append(Arrays.toString(co.tArgs()).replaceAll("[\\[\\]]", "")).append(");\n");
                }
                val inputs = co.inputArguments();
                val outputs = co.outputArguments();
                if(inputs != null ) {
                    sb.append("INDArray[] inputs = new INDArray[").append(inputs.size()).append("];\n");
                    for (int i = 0; i < inputs.size(); i++) {
                        sb.append("inputs[").append(i).append("] = ");
                        sb.append(createString(inputs.get(i)))
                                .append(";\n");
                    }
                    sb.append("op.addInputArgument(inputs);\n");
                }
                if(outputs != null ) {
                    sb.append("INDArray[] outputs = new INDArray[").append(outputs.size()).append("];\n");
                    for (int i = 0; i < outputs.size(); i++) {
                        sb.append("outputs[").append(i).append("] = ");
                        sb.append(createString(outputs.get(i)))
                                .append(";\n");
                    }
                    sb.append("op.addOutputArgument(outputs);\n");
                }
            } else {
                sb.append("Op op = new ").append(op.getClass().getName()).append("();\n");
                if(lOp.x() != null) {
                    sb.append("op.setX(").append(createString(lOp.x())).append(");\n");
                }
                if(lOp.y() != null) {
                    sb.append("op.setY(").append(createString(lOp.y())).append(");\n");
                }
                if(lOp.z() != null) {
                    sb.append("op.setZ").append(createString(lOp.z())).append(");\n");
                }
                if(lOp instanceof ScalarOp){
                    INDArray scalar = ((ScalarOp)lOp).scalar();
                    if(scalar != null){
                        sb.append("((ScalarOp)op).setScalar(").append(createString(scalar)).append(");\n");
                    }
                }
            }
            sb.append("Nd4j.exec(op);\n");
        }

        System.out.print(sb.toString());
    }

    private static String createString(INDArray arr){
        StringBuilder sb = new StringBuilder();

        if(arr.isEmpty()){
            sb.append("Nd4j.empty(DataType.").append(arr.dataType()).append(");");
        } else {
            sb.append("Nd4j.createFromArray(");

            DataType dt = arr.dataType();
            switch (dt){
                case DOUBLE:
                    double[] dArr = arr.dup().data().asDouble();
                    sb.append(Arrays.toString(dArr).replaceAll("[\\[\\]]", ""));
                    break;
                case FLOAT:
                case HALF:
                case BFLOAT16:
                    float[] fArr = arr.dup().data().asFloat();
                    sb.append(Arrays.toString(fArr)
                            .replaceAll(",", "f,")
                            .replaceAll("]", "f")
                            .replaceAll("[\\[\\]]", ""));
                    break;
                case LONG:
                case UINT32:
                case UINT64:
                    long[] lArr = arr.dup().data().asLong();
                    sb.append(Arrays.toString(lArr)
                            .replaceAll(",", "L,")
                            .replaceAll("]", "L")
                            .replaceAll("[\\[\\]]", ""));
                    break;
                case INT:
                case SHORT:
                case UBYTE:
                case BYTE:
                case UINT16:
                case BOOL:
                    int[] iArr = arr.dup().data().asInt();
                    sb.append(Arrays.toString(iArr).replaceAll("[\\[\\]]", ""));
                    break;
                case UTF8:
                    break;
                case COMPRESSED:
                case UNKNOWN:
                    break;
            }

            sb.append(").reshape(").append(Arrays.toString(arr.shape()).replaceAll("[\\[\\]]", ""))
                    .append(")");

            if(dt == DataType.HALF || dt == DataType.BFLOAT16 || dt == DataType.UINT32 || dt == DataType.UINT64 ||
                    dt == DataType.SHORT || dt == DataType.UBYTE || dt == DataType.BYTE || dt == DataType.UINT16 || dt == DataType.BOOL){
                sb.append(".cast(DataType.").append(arr.dataType()).append(")");
            }
        }

        return sb.toString();
    }

}
