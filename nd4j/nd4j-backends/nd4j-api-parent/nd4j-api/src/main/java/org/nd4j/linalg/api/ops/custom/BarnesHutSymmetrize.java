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

package org.nd4j.linalg.api.ops.custom;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

public class BarnesHutSymmetrize extends DynamicCustomOp {

    private INDArray output;
    private INDArray outCols;

    public BarnesHutSymmetrize(){ }

    public BarnesHutSymmetrize(INDArray rowP, INDArray colP, INDArray valP, long N,
                               INDArray outRows) {

        INDArray rowCounts = Nd4j.create(N);
        for (int n = 0; n < N; n++) {
            int begin = rowP.getInt(n);
            int end = rowP.getInt(n + 1);
            for (int i = begin; i < end; i++) {
                boolean present = false;
                for (int m = rowP.getInt(colP.getInt(i)); m < rowP.getInt(colP.getInt(i) + 1); m++) {
                    if (colP.getInt(m) == n) {
                        present = true;
                    }
                }
                if (present)
                    rowCounts.putScalar(n, rowCounts.getDouble(n) + 1);

                else {
                    rowCounts.putScalar(n, rowCounts.getDouble(n) + 1);
                    rowCounts.putScalar(colP.getInt(i), rowCounts.getDouble(colP.getInt(i)) + 1);
                }
            }
        }
        int outputCols = rowCounts.sum(Integer.MAX_VALUE).getInt(0);
        output = Nd4j.create(1, outputCols);
        outCols = Nd4j.create(new int[]{1, outputCols}, DataType.INT);

        inputArguments.add(rowP);
        inputArguments.add(colP);
        inputArguments.add(valP);

        outputArguments.add(outRows);
        outputArguments.add(outCols);
        outputArguments.add(output);

        iArguments.add(N);
    }

    public INDArray getSymmetrizedValues() {
        return output;
    }

    public INDArray getSymmetrizedCols() {
        return outCols;
    }

    @Override
    public String opName() {
        return "barnes_symmetrized";
    }
}
