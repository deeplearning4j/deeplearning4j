package org.datavec.python;

import org.datavec.api.writable.Writable;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.*;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class PythonTransformProcess implements Serializable{

    private Schema initialSchema;
    private Schema finalSchema;
    private PythonTransform pythonTransform;


    private PythonVariables schemaToPythonVariables(Schema schema) throws Exception{

        PythonVariables pyVars = new PythonVariables();
        int numCols = schema.numColumns();
        for (int i=0; i<numCols; i++){
            String colName = schema.getName(i);
            ColumnType colType = schema.getType(i);
            switch (colType){
                case Long:
                case Integer:
                    pyVars.addInt(colName);
                    break;
                case Double:
                case Float:
                    pyVars.addFloat(colName);
                    break;
                case String:
                    pyVars.addStr(colName);
                    break;
                case NDArray:
                    pyVars.addNDArray(colName);
                    break;
                default:
                    throw new Exception("Unsupported python input type: " + colType.toString());
            }
        }
        return pyVars;
    }

   public PythonTransformProcess(String pythonCode, Schema initialSchema, Schema finalSchema) throws Exception{
       this.initialSchema = initialSchema;
       this.finalSchema = finalSchema;

       PythonVariables pyInputs = schemaToPythonVariables(initialSchema);
       PythonVariables pyOutputs = schemaToPythonVariables(finalSchema);

       this.pythonTransform = new PythonTransform("default_transform", pythonCode, pyInputs, pyOutputs);

   }

   private PythonVariables getPyInputsFromWritables(List<Writable> writables) throws Exception{
        PythonVariables pyInputs = pythonTransform.getInputs();
        PythonVariables ret = new PythonVariables();

        for (String name: pyInputs.getVariables()){
            int colIdx = initialSchema.getIndexOfColumn(name);
            Writable w = writables.get(colIdx);
            PythonVariables.Type pyType = pyInputs.getType(name);
            switch (pyType){
                case INT:
                    ret.addInt(name, ((LongWritable)w).get());
                    break;
                case FLOAT:
                    ret.addFloat(name, ((DoubleWritable)w).get());
                    break;
                case STR:
                    ret.addStr(name, ((Text)w).toString());
                    break;
                case NDARRAY:
                    ret.addNDArray(name,((NDArrayWritable)w).get());
                    break;
            }

        }
        return ret;
   }

   private List<Writable> getWritablesFromPyOutputs(PythonVariables pyOuts){
        List<Writable> out = new ArrayList<>();
        for (int i=0; i<finalSchema.numColumns(); i++){
            String name = finalSchema.getName(i);
            PythonVariables.Type pyType = pythonTransform.getOutputs().getType(name);
            switch (pyType){
                case INT:
                    out.add((Writable) new LongWritable(pyOuts.getIntValue(name)));
                    break;
                case FLOAT:
                    out.add((Writable) new DoubleWritable(pyOuts.getFloatValue(name)));
                    break;
                case STR:
                    out.add((Writable) new Text(pyOuts.getStrValue(name)));
                    break;
                case NDARRAY:
                out.add((Writable) new NDArrayWritable(pyOuts.getNDArrayValue(name).getND4JArray()));
                break;
            }
        }
        return out;
   }

   public List<Writable> execute(List<Writable> inputs) throws Exception{
        PythonVariables pyInputs = getPyInputsFromWritables(inputs);
        PythonVariables pyOuts = PythonExecutioner.getInstance().exec(pythonTransform, pyInputs);
        List<Writable> out = getWritablesFromPyOutputs(pyOuts);
        return out;
   }


}
