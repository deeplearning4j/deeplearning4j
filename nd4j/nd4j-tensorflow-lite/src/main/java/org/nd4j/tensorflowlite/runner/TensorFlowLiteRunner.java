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
package org.nd4j.tensorflowlite.runner;

import java.io.Closeable;
import java.util.LinkedHashMap;
import java.util.Map;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.*;
import org.bytedeco.tensorflowlite.*;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.tensorflowlite.util.TFLiteUtils;

import static org.bytedeco.tensorflowlite.global.tensorflowlite.*;
import static org.nd4j.tensorflowlite.util.TFLiteUtils.*;

@Slf4j
public class TensorFlowLiteRunner implements Closeable  {
    private FlatBufferModel model;
    private BuiltinOpResolver resolver;
    private InterpreterBuilder builder;
    private Interpreter interpreter;

    @Builder
    public TensorFlowLiteRunner(String modelUri) {
        // Load model
        model = FlatBufferModel.BuildFromFile(modelUri);
        if (model == null || model.isNull()) {
            throw new RuntimeException("Cannot load " + modelUri);
        }
        //retain the model reference to prevent pre emptive release of the model.
        model.retainReference();

        // Build the interpreter with the InterpreterBuilder.
        // Note: all Interpreters should be built with the InterpreterBuilder,
        // which allocates memory for the Interpreter and does various set up
        // tasks so that the Interpreter can read the provided model.
        resolver = new BuiltinOpResolver();
        builder = new InterpreterBuilder(model, resolver);
        interpreter = new Interpreter((Pointer)null);
        builder.apply(interpreter);
        if (interpreter == null || interpreter.isNull()) {
            throw new RuntimeException("Cannot build interpreter for " + modelUri);
        }
        resolver.retainReference();
        builder.retainReference();
        interpreter.retainReference();

        // Allocate tensor buffers.
        if (interpreter.AllocateTensors() != kTfLiteOk) {
            throw new RuntimeException("Cannot allocate tensors for " + modelUri);
        }
        if (log.isInfoEnabled()) {
            log.info("=== Pre-invoke Interpreter State ===");
            PrintInterpreterState(interpreter);
        }
    }

    @Override
    public void close() {
        if (interpreter != null) {
            interpreter.releaseReference();
        }
        if (builder != null) {
            builder.releaseReference();
        }
        if (resolver != null) {
            resolver.releaseReference();
        }
        if (model != null) {
            model.releaseReference();
        }
    }

    /**
     * Execute the {@link #session}
     * using the given input {@link Map}
     * input
     * @param input the input map
     * @return a map of the names of the ndarrays
     */
    public Map<String,INDArray> exec(Map<String,INDArray> input) {
        long numInputNodes = interpreter.inputs().capacity();
        long numOutputNodes = interpreter.outputs().capacity();

        // Fill input buffers
        for (int i = 0; i < numInputNodes; i++) {
            BytePointer inputName = interpreter.GetInputName(i);
            INDArray arr = input.get(inputName.getString());
            TfLiteTensor inputTensor = interpreter.input_tensor(i);
            Preconditions.checkState(inputTensor != null,"Input must be a tensor.");
            Nd4j.copy(arr, getArray(inputTensor));
        }

        // Run inference
        if (interpreter.Invoke() != kTfLiteOk) {
            throw new RuntimeException("Cannot invoke interpreter for " + model);
        }
        if (log.isInfoEnabled()) {
            log.info("=== Post-invoke Interpreter State ===");
            PrintInterpreterState(interpreter);
        }

        Map<String, INDArray> ret = new LinkedHashMap<>();

        // Read output buffers
        for (int i = 0; i < numOutputNodes; i++) {
            TfLiteTensor outputTensor = interpreter.output_tensor(i);
            ret.put(interpreter.GetOutputName(i).getString(), getArray(outputTensor));
        }
        return ret;
    }
}
