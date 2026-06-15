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

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Hexagon NPU-specific operation context.
 *
 * Holds input/output arrays, arguments, and execution state for Hexagon NPU operations.
 */
public class HexagonOpContext implements OpContext {

    private final List<INDArray> inputArrays = new ArrayList<>();
    private final List<INDArray> outputArrays = new ArrayList<>();
    private final List<Double> tArguments = new ArrayList<>();
    private final List<Long> iArguments = new ArrayList<>();
    private final List<Boolean> bArguments = new ArrayList<>();
    private final List<DataType> dArguments = new ArrayList<>();
    private final List<Integer> axisArguments = new ArrayList<>();

    private int rngSeed = 0;

    public HexagonOpContext() {
    }

    @Override
    public void setIArguments(long... iArgs) {
        iArguments.clear();
        for (long arg : iArgs) {
            iArguments.add(arg);
        }
    }

    @Override
    public List<Long> getIArguments() {
        return iArguments;
    }

    @Override
    public int numIArguments() {
        return iArguments.size();
    }

    @Override
    public void setTArguments(double... tArgs) {
        tArguments.clear();
        for (double arg : tArgs) {
            tArguments.add(arg);
        }
    }

    @Override
    public List<Double> getTArguments() {
        return tArguments;
    }

    @Override
    public int numTArguments() {
        return tArguments.size();
    }

    @Override
    public void setBArguments(boolean... bArgs) {
        bArguments.clear();
        for (boolean arg : bArgs) {
            bArguments.add(arg);
        }
    }

    @Override
    public List<Boolean> getBArguments() {
        return bArguments;
    }

    @Override
    public int numBArguments() {
        return bArguments.size();
    }

    @Override
    public void setDArguments(DataType... dArgs) {
        dArguments.clear();
        for (DataType arg : dArgs) {
            dArguments.add(arg);
        }
    }

    @Override
    public List<DataType> getDArguments() {
        return dArguments;
    }

    @Override
    public int numDArguments() {
        return dArguments.size();
    }

    @Override
    public void setAxisArguments(int... axisArgs) {
        axisArguments.clear();
        for (int arg : axisArgs) {
            axisArguments.add(arg);
        }
    }

    @Override
    public List<Integer> getAxisArguments() {
        return axisArguments;
    }

    @Override
    public int numAxisArguments() {
        return axisArguments.size();
    }

    @Override
    public void setInputArrays(List<INDArray> inputs) {
        inputArrays.clear();
        inputArrays.addAll(inputs);
    }

    @Override
    public void setInputArrays(INDArray... inputs) {
        inputArrays.clear();
        for (INDArray input : inputs) {
            inputArrays.add(input);
        }
    }

    @Override
    public void addInputArgument(INDArray input) {
        inputArrays.add(input);
    }

    @Override
    public List<INDArray> getInputArrays() {
        return inputArrays;
    }

    @Override
    public int numInputArguments() {
        return inputArrays.size();
    }

    @Override
    public INDArray getInputArray(int i) {
        return inputArrays.get(i);
    }

    @Override
    public void setOutputArrays(List<INDArray> outputs) {
        outputArrays.clear();
        outputArrays.addAll(outputs);
    }

    @Override
    public void setOutputArrays(INDArray... outputs) {
        outputArrays.clear();
        for (INDArray output : outputs) {
            outputArrays.add(output);
        }
    }

    @Override
    public void addOutputArgument(INDArray output) {
        outputArrays.add(output);
    }

    @Override
    public List<INDArray> getOutputArrays() {
        return outputArrays;
    }

    @Override
    public int numOutputArguments() {
        return outputArrays.size();
    }

    @Override
    public INDArray getOutputArray(int i) {
        return outputArrays.get(i);
    }

    @Override
    public void setRngStates(long rootState, long nodeState) {
        // Hexagon NPU uses different RNG mechanism
    }

    @Override
    public long[] getRngStates() {
        return new long[2];
    }

    @Override
    public void setRandom(org.nd4j.linalg.api.rng.Random rng) {
        if (rng != null) {
            this.rngSeed = (int) rng.getSeed();
        }
    }

    @Override
    public org.nd4j.linalg.api.rng.Random getRandom() {
        return Nd4j.getRandom();
    }

    @Override
    public void purge() {
        inputArrays.clear();
        outputArrays.clear();
        tArguments.clear();
        iArguments.clear();
        bArguments.clear();
        dArguments.clear();
        axisArguments.clear();
    }

    @Override
    public void close() {
        purge();
    }

    @Override
    public void transferTArguments() {
        // Transfer T arguments to native context
    }

    @Override
    public void allowHelpers(boolean allowed) {
        // Hexagon always uses platform helpers
    }

    @Override
    public boolean helpersAllowed() {
        return true;
    }

    @Override
    public void shapeFunctionOverride(boolean enabled) {
        // Not used for Hexagon
    }

    @Override
    public boolean isShapeFunctionOverride() {
        return false;
    }

    @Override
    public void setInputArray(int index, INDArray arr) {
        while (inputArrays.size() <= index) {
            inputArrays.add(null);
        }
        inputArrays.set(index, arr);
    }

    @Override
    public void setOutputArray(int index, INDArray arr) {
        while (outputArrays.size() <= index) {
            outputArrays.add(null);
        }
        outputArrays.set(index, arr);
    }
}
