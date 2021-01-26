/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

package org.datavec.python.keras;

import org.datavec.python.Python;
import org.datavec.python.PythonException;
import org.datavec.python.PythonObject;
import org.datavec.python.PythonProcess;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Model {

    private PythonObject pyModel;


    private static PythonObject installAndImportTF() throws PythonException{
        if (!PythonProcess.isPackageInstalled("tensorflow")){
            PythonProcess.pipInstall("tensorflow");
        }
        return Python.importModule("tensorflow");
    }
    private static PythonObject getKerasModule() throws PythonException{
        PythonObject tf = installAndImportTF();
        PythonObject keras = tf.attr("keras");
        tf.del();
        return keras;
    }

    private static PythonObject loadModel(String s) throws PythonException{
        PythonObject models = getKerasModule().attr("models");
        PythonObject loadModelF = models.attr("load_model");
        PythonObject model = loadModelF.call(s);
        models.del();
        loadModelF.del();
        return model;
    }

    public Model(String path) throws PythonException{
        pyModel = loadModel(path);
    }

    public INDArray[] predict(INDArray... inputs) throws PythonException{
        PythonObject predictF = pyModel.attr("predict");
        PythonObject inputList = new PythonObject(inputs);
        PythonObject pyOut = predictF.call(inputList);
        INDArray[] out;
        if (Python.isinstance(pyOut, Python.listType())){
            out = new INDArray[Python.len(pyOut).toInt()];
            for(int i = 0; i < out.length; i++){
                out[i] = pyOut.get(i).toNumpy().getNd4jArray();
            }
        }
        else{
            out = new INDArray[]{
                    pyOut.toNumpy().getNd4jArray()};
            }

        predictF.del();
        inputList.del();
        pyOut.del();
        return out;
    }

    public int numInputs(){
        PythonObject inputs = pyModel.attr("inputs");
        PythonObject pyNumInputs = Python.len(inputs);
        int ret = pyNumInputs.toInt();
        inputs.del();
        pyNumInputs.del();
        return ret;
    }
    public int numOutputs(){
        PythonObject outputs = pyModel.attr("outputs");
        PythonObject pyNumOutputs = Python.len(outputs);
        int ret = pyNumOutputs.toInt();
        outputs.del();
        pyNumOutputs.del();
        return ret;
    }

    public long[][] inputShapes(){
        long[][] ret = new long[numInputs()][];
        for (int i = 0; i < ret.length; i++){
            ret[i] = inputShapeAt(i);
        }
        return ret;
    }

    public long[][] outputShapes(){
        long[][] ret = new long[numOutputs()][];
        for (int i = 0; i < ret.length; i++){
            ret[i] = outputShapeAt(i);
        }
        return ret;
    }

    public long[] inputShapeAt(int input){
        PythonObject inputs = pyModel.attr("inputs");
        PythonObject tensor = inputs.get(input);
        PythonObject tensorShape = tensor.attr("shape");
        PythonObject shapeList = Python.list(tensorShape);
        PythonObject pyNdim = Python.len(shapeList);
        int ndim = pyNdim.toInt();
        long[] shape = new long[ndim];
        for(int i = 0; i < shape.length; i++){
            PythonObject pyDim = shapeList.get(i);
            if (pyDim == null || !Python.isinstance(pyDim, Python.intType())){
                shape[i] = -1;
            }
            else{
                shape[i] = pyDim.toLong();
            }
        }
        pyNdim.del();
        shapeList.del();
        tensorShape.del();
        tensor.del();
        inputs.del();
        return shape;
    }

    public long[] outputShapeAt(int output){
        PythonObject inputs = pyModel.attr("outputs");
        PythonObject tensor = inputs.get(output);
        PythonObject tensorShape = tensor.attr("shape");
        PythonObject shapeList = Python.list(tensorShape);
        PythonObject pyNdim = Python.len(shapeList);
        int ndim = pyNdim.toInt();
        long[] shape = new long[ndim];
        for(int i = 0; i < shape.length; i++){
            PythonObject pyDim = shapeList.get(i);
            if (pyDim == null || !Python.isinstance(pyDim, Python.intType())){
                shape[i] = -1;
            }
            else{
                shape[i] = pyDim.toLong();
            }
        }
        pyNdim.del();
        shapeList.del();
        tensorShape.del();
        tensor.del();
        inputs.del();
        return shape;
    }
}
