package org.datavec.python.keras;

import org.datavec.python.Python;
import org.datavec.python.PythonException;
import org.datavec.python.PythonObject;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Model {

    private PythonObject pyModel;

    private static PythonObject getKerasModule() throws PythonException{
        PythonObject tf = Python.importModule("tensorflow");
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
