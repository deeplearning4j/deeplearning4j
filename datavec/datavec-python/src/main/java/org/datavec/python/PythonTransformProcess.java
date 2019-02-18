package org.datavec.python;


import com.sun.corba.se.spi.ior.Writeable;
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
                case String:
                    pyVars.addStr(colName);
                case NDArray:
                    pyVars.addNDArray(colName);
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

   private void setPyInputs(List<Writeable> writeables) throws Exception{
        PythonVariables pyInputs = pythonTransform.getInputs();
        for (String name: pyInputs.getVariables()){
            int colIdx = initialSchema.getIndexOfColumn(name);
            Writeable w = writeables.get(colIdx);
            PythonVariables.Type pyType = pyInputs.getType(name);
            switch (pyType){
                case INT:
                    pyInputs.setValue(name, ((LongWritable)w).get());
                    break;
                case FLOAT:
                    pyInputs.setValue(name, ((DoubleWritable)w).get());
                    break;
                case STR:
                    pyInputs.setValue(name, ((Text)w).toString());
                    break;
                case NDARRAY:
                    pyInputs.setValue(name,((NDArrayWritable)w).get());
                    break;
            }

        }
   }

   private List<Writeable> getPyOutputs(){
        PythonVariables pyOuts = pythonTransform.getOutputs();
        List<Writeable> out = new ArrayList<>();
        for (int i=0; i<finalSchema.numColumns(); i++){
            String name = finalSchema.getName(i);
            PythonVariables.Type pyType = pyOuts.getType(name);
            switch (pyType){
                case INT:
                    out.add((Writeable) new LongWritable(pyOuts.getIntValue(name)));
                    break;
                case FLOAT:
                    out.add((Writeable) new DoubleWritable(pyOuts.getFloatValue(name)));
                    break;
                case NDARRAY:
                out.add((Writeable) new NDArrayWritable(pyOuts.getNDArrayValue(name).getND4JArray()));
                break;
            }
        }
        return out;
   }

   public List<Writeable> execute(List<Writeable> inputs) throws Exception{
        setPyInputs(inputs);
        PythonExecutioner.getInstance().safeExec(pythonTransform);
        return getPyOutputs();
   }


}
