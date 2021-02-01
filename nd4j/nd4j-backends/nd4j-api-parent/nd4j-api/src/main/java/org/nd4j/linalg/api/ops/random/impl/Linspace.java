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

package org.nd4j.linalg.api.ops.random.impl;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;

/**
 * Linspace/arange Op implementation, generates from..to distribution within Z
 *
 * @author raver119@gmail.com
 */
public class Linspace extends BaseRandomOp {
    private double from;
    private double to;
    private double step;
    private long length;

    public Linspace() {
        // no-op
    }

    public Linspace(double from, long length, double step, DataType dataType){
        this(Nd4j.createUninitialized(dataType, new long[] {length}, Nd4j.order()), from, from, step);
    }

    public Linspace(double from, double to, long length, DataType dataType) {
        this(Nd4j.createUninitialized(dataType, new long[] {length}, Nd4j.order()), from, to);
    }

    public Linspace(@NonNull INDArray z, double from, double to) {
        super(null, null, z);
        this.from = from;
        this.to = to;
        this.length = z.length();
        double step = 0.0;
        this.extraArgs = new Object[] {from, to, step};
    }

    public Linspace(@NonNull INDArray z, double from, double to, double step) {
        super(null, null, z);
        this.from = from;
        this.to = to;
        this.length = z.length();
        this.step = step;
        this.extraArgs = new Object[] {from, to, step};
    }

    public Linspace(SameDiff sd, double from, double to, long length){
        super(sd, new long[]{length});
        this.sameDiff = sd;
        this.from = from;
        this.to = to;
        this.length = length;
        double step = 0.0; //(to - from) / (length - 1);
        this.extraArgs = new Object[] {from, to, step};
    }

    @Override
    public int opNum() {
        return 4;
    }

    @Override
    public String opName(){
        return "linspace_random";
    }

    @Override
    public INDArray x(){
        //Workaround/hack for: https://github.com/deeplearning4j/deeplearning4j/issues/6723
        //If x or y is present, can't execute this op properly (wrong signature is used)
        return null;
    }

    @Override
    public INDArray y(){
        //Workaround/hack for: https://github.com/deeplearning4j/deeplearning4j/issues/6723
        //If x or y is present, can't execute this op properly (wrong signature is used)
        return null;
    }

    @Override
    public void setX(INDArray x){
        //Workaround/hack for: https://github.com/deeplearning4j/deeplearning4j/issues/6723
        //If x or y is present, can't execute this op properly (wrong signature is used)
        this.x = null;
    }

    @Override
    public void setY(INDArray y){
        //Workaround for: https://github.com/deeplearning4j/deeplearning4j/issues/6723
        //If x or y is present, can't execute this op properly (wrong signature is used)
        this.y = null;
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        return Collections.singletonList(LongShapeDescriptor.fromShape(new long[]{length}, DataType.FLOAT));      //TODO Don't hardcode float!
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //No inputs
        return Collections.emptyList();
    }
}
