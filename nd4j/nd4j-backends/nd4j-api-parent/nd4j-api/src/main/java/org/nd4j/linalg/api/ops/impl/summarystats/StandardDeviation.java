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

package org.nd4j.linalg.api.ops.impl.summarystats;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.bp.StandardDeviationBp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Standard deviation (sqrt of variance)
 *
 * @author Adam Gibson
 */
public class StandardDeviation extends Variance {
    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, boolean biasCorrected, boolean keepDims, int[] dimensions) {
        super(sameDiff, i_v, biasCorrected, keepDims, dimensions );
    }

    public StandardDeviation(INDArray x, boolean biasCorrected, boolean keepDims, int... dimension) {
        super(x, biasCorrected, dimension);
        this.keepDims = keepDims;
    }

    public StandardDeviation(INDArray x, boolean biasCorrected, int... dimension) {
        super(x, biasCorrected, dimension);
    }


    public StandardDeviation() {
    }

    public StandardDeviation(INDArray x) {
        super(x);
    }

    public StandardDeviation(INDArray x, INDArray z, boolean biasCorrected, int... dimension) {
        super(x, z, biasCorrected, dimension);
    }

    public StandardDeviation(INDArray x, INDArray z, boolean newFormat, boolean keepDims, int[] dimensions) {
        super(x, z, newFormat, keepDims, dimensions);
    }

    @Override
    public int opNum() {
        return 1;
    }

    @Override
    public String opName() {
        return "std";
    }

    @Override
    public String onnxName(){
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName(){
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }

    @Override
    public Type getOpType() {
        return Type.SUMMARYSTATS;
    }

    @Override
    public Type opType(){
        return Type.SUMMARYSTATS;
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        //Here: calculating dL/dIn given dL/dOut (i.e., i_v1) and input/output
        //If out = stdev(in) then:
        //dL/dIn = dL/dOut * dOut/dIn
        //dOut/dIn_i = (in_i-mean)/(stdev * (n-1))
        return new StandardDeviationBp(sameDiff, arg(), grad.get(0), biasCorrected, keepDims, dimensions).outputs();
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        if(args().length < 1) {
            throw new ND4JIllegalStateException("Unable to compute input shape. No arguments found.");
        }

        long[] argShape = arg().getShape();
        if (argShape == null && x() == null) {
            return Collections.emptyList();
        }
        long[] inputShape = (argShape == null || Shape.isPlaceholderShape(argShape) ? x().shape() : argShape);

        val ret = new ArrayList<LongShapeDescriptor>(1);
        val reducedShape = Shape.getReducedShape(inputShape,dimensions, isKeepDims());
        ret.add(LongShapeDescriptor.fromShape(reducedShape, resultType()));
        return ret;
    }
}
