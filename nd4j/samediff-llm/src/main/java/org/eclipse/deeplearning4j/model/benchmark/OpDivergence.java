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

package org.eclipse.deeplearning4j.model.benchmark;

import java.util.Arrays;

/**
 * Records a single op-level divergence between reference and test execution paths.
 *
 * <p>Created by {@link DspAccuracyValidator#compareArrays} when two output arrays
 * differ beyond the configured tolerance. Contains enough information to trace
 * the divergence back to its source: the op name, step index, the exact element
 * with maximum difference, and the reference/test values at that element.</p>
 */
public class OpDivergence {

    private final int stepIdx;
    private final String opName;
    private final String varName;
    private final long[] shape;
    private final String dataType;
    private final double maxAbsDiff;
    private final double meanAbsDiff;
    private final double maxRelDiff;
    private final long maxDiffIndex;
    private final double referenceValueAtMax;
    private final double testValueAtMax;
    private final String[] inputVarNames;
    private final double absTolUsed;

    public OpDivergence(int stepIdx, String opName, String varName,
                        long[] shape, String dataType,
                        double maxAbsDiff, double meanAbsDiff, double maxRelDiff,
                        long maxDiffIndex, double referenceValueAtMax, double testValueAtMax,
                        String[] inputVarNames, double absTolUsed) {
        this.stepIdx = stepIdx;
        this.opName = opName;
        this.varName = varName;
        this.shape = shape;
        this.dataType = dataType;
        this.maxAbsDiff = maxAbsDiff;
        this.meanAbsDiff = meanAbsDiff;
        this.maxRelDiff = maxRelDiff;
        this.maxDiffIndex = maxDiffIndex;
        this.referenceValueAtMax = referenceValueAtMax;
        this.testValueAtMax = testValueAtMax;
        this.inputVarNames = inputVarNames;
        this.absTolUsed = absTolUsed;
    }

    public int getStepIdx() { return stepIdx; }
    public String getOpName() { return opName; }
    public String getVarName() { return varName; }
    public long[] getShape() { return shape; }
    public String getDataType() { return dataType; }
    public double getMaxAbsDiff() { return maxAbsDiff; }
    public double getMeanAbsDiff() { return meanAbsDiff; }
    public double getMaxRelDiff() { return maxRelDiff; }
    public long getMaxDiffIndex() { return maxDiffIndex; }
    public double getReferenceValueAtMax() { return referenceValueAtMax; }
    public double getTestValueAtMax() { return testValueAtMax; }
    public String[] getInputVarNames() { return inputVarNames; }
    public double getAbsTolUsed() { return absTolUsed; }

    @Override
    public String toString() {
        return String.format("Step %4d %-25s %-40s maxAbs=%.6e meanAbs=%.6e maxRel=%.6e " +
                        "(tol=%.2e) ref=%.6e test=%.6e idx=%d shape=%s dtype=%s",
                stepIdx, opName, varName, maxAbsDiff, meanAbsDiff, maxRelDiff,
                absTolUsed, referenceValueAtMax, testValueAtMax, maxDiffIndex,
                Arrays.toString(shape), dataType);
    }
}
