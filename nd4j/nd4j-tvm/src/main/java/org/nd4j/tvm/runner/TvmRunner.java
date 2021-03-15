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
package org.nd4j.tvm.runner;

import java.io.Closeable;
import java.util.LinkedHashMap;
import java.util.Map;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.*;
import org.bytedeco.tvm.*;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.bytedeco.tvm.global.tvm_runtime.*;
import static org.nd4j.tvm.util.TVMUtils.*;

@Slf4j
public class TvmRunner implements Closeable  {
    private static DLContext ctx;
    private  org.bytedeco.tvm.Module modFactory;
    private TVMValue values;
    private IntPointer codes;
    private TVMArgsSetter setter;
    private TVMRetValue rv;
    private org.bytedeco.tvm.Module gmod;
    private PackedFunc getNumInputs;
    private PackedFunc getNumOutputs;
    private PackedFunc setInput;
    private PackedFunc getOutput;
    private PackedFunc run;

    @Builder
    public TvmRunner(String modelUri) {
        if (ctx == null) {
            ctx = new DLContext().device_type(kDLCPU).device_id(0);
            ctx.retainReference();
        }

        // create the runtime module
        try (PointerScope scope = new PointerScope()) {
            modFactory = org.bytedeco.tvm.Module.LoadFromFile(modelUri);
            values = new TVMValue(2);
            codes = new IntPointer(2);
            setter = new TVMArgsSetter(values, codes);
            setter.apply(0, ctx);
            rv = new TVMRetValue();
            modFactory.GetFunction("default").CallPacked(new TVMArgs(values, codes, 1), rv);
            gmod = rv.asModule();
            getNumInputs = gmod.GetFunction("get_num_inputs");
            getNumOutputs = gmod.GetFunction("get_num_outputs");
            setInput = gmod.GetFunction("set_input");
            getOutput = gmod.GetFunction("get_output");
            run = gmod.GetFunction("run");
            // retain the session reference to prevent pre emptive release of the session.
            modFactory.retainReference();
            values.retainReference();
            codes.retainReference();
            setter.retainReference();
            rv.retainReference();
            gmod.retainReference();
            getNumInputs.retainReference();
            getNumOutputs.retainReference();
            setInput.retainReference();
            getOutput.retainReference();
            run.retainReference();
        }
    }

    @Override
    public void close() {
        if (run != null) {
            run.releaseReference();
        }
        if (getOutput != null) {
            getOutput.releaseReference();
        }
        if (setInput != null) {
            setInput.releaseReference();
        }
        if (getNumOutputs != null) {
            getNumOutputs.releaseReference();
        }
        if (getNumInputs != null) {
            getNumInputs.releaseReference();
        }
        if (gmod != null) {
            gmod.releaseReference();
        }
        if (rv != null) {
            rv.releaseReference();
        }
        if (setter != null) {
            setter.releaseReference();
        }
        if (codes != null) {
            codes.releaseReference();
        }
        if (values != null) {
            values.releaseReference();
        }
        if (modFactory != null) {
            modFactory.releaseReference();
        }
    }

    /**
     * Execute the {@link #run} function
     * using the given input {@link Map}
     * @param input the input map
     * @return a map of the names of the ndarrays
     */
    public Map<String,INDArray> exec(Map<String,INDArray> input) {
        try (PointerScope scope = new PointerScope()) {
            getNumInputs.CallPacked(new TVMArgs(values, codes, 0), rv);
            long numInputNodes = rv.asLong();
            getNumOutputs.CallPacked(new TVMArgs(values, codes, 0), rv);
            long numOutputNodes = rv.asLong();

            // set the right input
            for (Map.Entry<String,INDArray> e : input.entrySet()) {
                String name = e.getKey();
                INDArray arr = e.getValue();
                DLTensor inputTensor = getTensor(arr, ctx);
                Preconditions.checkState(inputTensor != null,"Input must be a tensor.");
                setter.apply(0, new BytePointer(name));
                setter.apply(1, inputTensor);
                setInput.CallPacked(new TVMArgs(values, codes, 2), rv);
            }

            // run the code
            run.CallPacked(new TVMArgs(values, codes, 0), rv);

            Map<String, INDArray> ret = new LinkedHashMap<>();

            // get the output
            for (int i = 0; i < numOutputNodes; i++) {
                setter.apply(0, i);
                getOutput.CallPacked(new TVMArgs(values, codes, 1), rv);
                DLTensor outputTensor = rv.asDLTensor();
                ret.put(Integer.toString(i), getArray(outputTensor));
            }
            return ret;
        }
    }
}
