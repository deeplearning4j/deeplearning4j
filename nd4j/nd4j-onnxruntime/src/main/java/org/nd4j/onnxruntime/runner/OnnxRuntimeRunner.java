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
package org.nd4j.onnxruntime.runner;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import onnx.Onnx;
import org.apache.commons.io.FileUtils;
import org.bytedeco.javacpp.*;
import org.bytedeco.onnxruntime.*;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.onnxruntime.runner.enums.ONNXType;
import org.nd4j.onnxruntime.util.ONNXUtils;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.util.*;

import static org.bytedeco.onnxruntime.global.onnxruntime.*;
import static org.nd4j.onnxruntime.util.ONNXUtils.*;

@Slf4j
@Getter
public class OnnxRuntimeRunner implements Closeable  {

    private Session session;
    private RunOptions runOptions;
    private MemoryInfo memoryInfo;
    private AllocatorWithDefaultOptions allocator;
    private SessionOptions sessionOptions;
    private   static Env env;
    private Pointer bp;
    private Onnx.ModelProto modelProto;

    @Builder
    public OnnxRuntimeRunner(String modelUri) {
        if(env == null) {
            env = new Env(ONNXUtils.getOnnxLogLevelFromLogger(log), new BytePointer("nd4j-serving-onnx-session-" + UUID.randomUUID()));
            env.retainReference();
        }

        sessionOptions = new SessionOptions();
        sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);
        sessionOptions.retainReference();
        allocator = new AllocatorWithDefaultOptions();
        allocator.retainReference();
        if(modelUri != null) {
            bp = Loader.getPlatform().toLowerCase().startsWith("windows") ? new CharPointer(modelUri) : new BytePointer(modelUri);
            session = new Session(env, bp, sessionOptions);
            //retain the session reference to prevent pre emptive release of the session.
            session.retainReference();
            try {
                modelProto = Onnx.ModelProto.parseFrom(FileUtils.readFileToByteArray(new File(modelUri)));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        runOptions = new RunOptions();
        memoryInfo = MemoryInfo.CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);


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
    public Map<String,SDValue> execValues(Map<String, SDValue> input) {
        long numInputNodes = session.GetInputCount();
        long numOutputNodes = session.GetOutputCount();

        PointerPointer<BytePointer> inputNodeNames = new PointerPointer<>(numInputNodes);
        PointerPointer<BytePointer> outputNodeNames = new PointerPointer<>(numOutputNodes);

        Value inputVal = new Value(numInputNodes);
        for (int i = 0; i < numInputNodes; i++) {
            BytePointer inputName = session.GetInputName(i, allocator.asOrtAllocator());
            inputNodeNames.put(i, inputName);
            ONNXType typeForInput = getTypeForInput(session, i);
            List<INDArray> arr = input.get(inputName.getString()).getListValue();
            if(arr.size() == 1 && typeForInput == ONNXType.ONNX_TYPE_TENSOR) {
                INDArray arr2 = arr.get(0);
                Value inputTensor = getTensor(arr2, memoryInfo);
                Preconditions.checkState(inputTensor.IsTensor(),"Input must be a tensor.");
                inputVal.position(i).put(inputTensor);
            }
            //empty sequence
            else if(arr.size() == 0) {
                    throw new IllegalArgumentException("Onnx Runtime does not support empty sequences! Found at input name " + inputName.getString());
            } else if(arr.size() > 1 || typeForInput == ONNXType.ONNX_TYPE_SEQUENCE) {
                ValueVector inputTensor = getSequence(arr, memoryInfo);
                inputVal.position(i).put(Value.CreateSequence(inputTensor));
            }

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
        Map<String, SDValue> ret = new LinkedHashMap<>();

        for (int i = 0; i < numOutputNodes; i++) {
            Value outValue = outputVector.get(i);
            outValue.retainReference();
            if(outValue.IsTensor()) {
                INDArray arr = getArray(outValue);
                ret.put((outputNodeNames.get(BytePointer.class, i)).getString(), SDValue.create(arr));
            } else  {
                INDArray[] seq = ndarraysFromSequence(outValue,allocator.asOrtAllocator());
                ret.put((outputNodeNames.get(BytePointer.class, i)).getString(), SDValue.create(Arrays.asList(seq)));
            }

        }

        return ret;


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
            ONNXType typeForOutput = getTypeForOutput(session, i);
            switch(typeForOutput) {
                case ONNX_TYPE_SEQUENCE:
                    long count = outValue.GetCount();
                    break;
                case ONNX_TYPE_TENSOR:
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
                    break;
                case ONNX_TYPE_MAP:
                case ONNX_TYPE_OPAQUE:
                case ONNX_TYPE_UNKNOWN:
                case ONNX_TYPE_OPTIONAL:
                case ONNX_TYPE_SPARSE_TENSOR:
                default:
                    throw new IllegalStateException("Unable to get type " + typeForOutput + " only accepts tensors and sequences.");
            }

        }

        return ret;


    }


}
