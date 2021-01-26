/* ******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.nd4j.onnxruntime.runner;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.*;
import org.bytedeco.onnxruntime.*;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.onnxruntime.util.ONNXUtils;

import java.io.Closeable;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.bytedeco.onnxruntime.global.onnxruntime.*;
import static org.nd4j.onnxruntime.util.ONNXUtils.getDataBuffer;
import static org.nd4j.onnxruntime.util.ONNXUtils.getTensor;

@Slf4j
public class OnnxRuntimeRunner implements Closeable  {
    private Session session;
    private RunOptions runOptions;
    private MemoryInfo memoryInfo;
    private AllocatorWithDefaultOptions allocator;
    private SessionOptions sessionOptions;
    private   static Env env;
    private Pointer bp;


    @Builder
    public OnnxRuntimeRunner(String modelUri) {
        if(env == null) {
            env = new Env(ONNXUtils.getOnnxLogLevelFromLogger(log), new BytePointer("nd4j-serving-onnx-session-" + UUID.randomUUID().toString()));
            env.retainReference();
        }

        sessionOptions = new SessionOptions();
        sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.retainReference();
        allocator = new AllocatorWithDefaultOptions();
        allocator.retainReference();
        bp = Loader.getPlatform().toLowerCase().startsWith("windows") ? new CharPointer(modelUri) : new BytePointer(modelUri);
        runOptions = new RunOptions();
        memoryInfo = MemoryInfo.CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        session = new Session(env, bp, sessionOptions);
        //retain the session reference to prevent pre emptive release of the session.
        session.retainReference();

    }



    @Override
    public void close() {
        if(session != null) {
            session.close();
        }

        sessionOptions.releaseReference();
        allocator.releaseReference();
        runOptions.releaseReference();
    }


    /**
     * Execute the {@link #session}
     * using the given input {@link Map}
     * input
     * @param input the input map
     * @return a map of the names of the ndarrays
     */
    public Map<String,INDArray> exec(Map<String,INDArray> input) {
        long numInputNodes = session.GetInputCount();
        long numOutputNodes = session.GetOutputCount();

        PointerPointer<BytePointer> inputNodeNames = new PointerPointer<>(numInputNodes);
        PointerPointer<BytePointer> outputNodeNames = new PointerPointer<>(numOutputNodes);

        Value inputVal = new Value(numInputNodes);

        for (int i = 0; i < numInputNodes; i++) {
            BytePointer inputName = session.GetInputName(i, allocator.asOrtAllocator());
            inputNodeNames.put(i, inputName);
            INDArray arr = input.get(inputName.getString());
            Value inputTensor = getTensor(arr, memoryInfo);
            Preconditions.checkState(inputTensor.IsTensor(),"Input must be a tensor.");
            inputVal.position(i).put(inputTensor);
        }

        //reset position after iterating
        inputVal.position(0);



        for (int i = 0; i < numOutputNodes; i++) {
            BytePointer outputName = session.GetOutputName(i, allocator.asOrtAllocator());
            outputNodeNames.put(i, outputName);
        }

        ValueVector outputVector = session.Run(
                runOptions,
                inputNodeNames,
                inputVal,
                numInputNodes,
                outputNodeNames,
                numOutputNodes);

        outputVector.retainReference();
        Map<String, INDArray> ret = new LinkedHashMap<>();

        for (int i = 0; i < numOutputNodes; i++) {
            Value outValue = outputVector.get(i);
            outValue.retainReference();
            TypeInfo typeInfo = session.GetOutputTypeInfo(i);
            DataBuffer buffer = getDataBuffer(outValue);
            LongPointer longPointer = outValue.GetTensorTypeAndShapeInfo().GetShape();
            //shape info can be null
            if(longPointer != null) {
                long[] shape = new long[(int) longPointer.capacity()];
                longPointer.get(shape);
                ret.put((outputNodeNames.get(BytePointer.class, i)).getString(), Nd4j.create(buffer).reshape(shape));
            } else {
                ret.put((outputNodeNames.get(BytePointer.class, i)).getString(), Nd4j.create(buffer));

            }
        }

        return ret;


    }


}
