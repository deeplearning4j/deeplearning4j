package org.datavec.python;

import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.*;

import java.util.List;

public class PythonCondition implements Condition {

    private Schema inputSchema;
    private PythonVariables pyInputs;
    private PythonTransform pythonTransform;
    private String code;
    public PythonCondition(String pythonCode){
     code = pythonCode;
    }

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
    private PythonVariables getPyInputsFromWritables(List<Writable> writables){

        PythonVariables ret = new PythonVariables();

        for (String name: pyInputs.getVariables()){
            int colIdx = inputSchema.getIndexOfColumn(name);
            Writable w = writables.get(colIdx);
            PythonVariables.Type pyType = pyInputs.getType(name);
            switch (pyType){
                case INT:
                    if (w instanceof LongWritable){
                        ret.addInt(name, ((LongWritable)w).get());
                    }
                    else{
                        ret.addInt(name, ((IntWritable)w).get());
                    }

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
    @Override
    public void setInputSchema(Schema inputSchema){
        this.inputSchema = inputSchema;
        try{
            pyInputs = schemaToPythonVariables(inputSchema);
            PythonVariables pyOuts = new PythonVariables();
            pyOuts.addInt("out");
            pythonTransform = new PythonTransform(
                    code + "\n\nout=f()\nout=0 if out is None else int(out)", // TODO: remove int conversion after boolean support is covered
                    pyInputs,
                    pyOuts
            );
        }
        catch (Exception e){
            // FixMe: do not suppress this
            System.out.println(e);
        }



    }

    @Override
    public Schema getInputSchema(){
        return inputSchema;
    }

    public String[] outputColumnNames(){
        String[] columnNames = new String[inputSchema.numColumns()];
        inputSchema.getColumnNames().toArray(columnNames);
        return columnNames;
    }

    public String outputColumnName(){
        return outputColumnNames()[0];
    }

    public String[] columnNames(){
        return outputColumnNames();
    }

    public String columnName(){
        return outputColumnName();
    }

    public Schema transform(Schema inputSchema){
        return inputSchema;
    }

    public boolean condition(List<Writable> list){
        PythonVariables inputs = getPyInputsFromWritables(list);
        try{
            PythonVariables output = PythonExecutioner.getInstance().exec(pythonTransform, inputs);
            boolean ret = output.getIntValue("out") != 0;
            return ret;
        }
        catch (Exception e){
            System.out.println(e);
        }
        return true;

    }

    public boolean condition(Object input){
        return condition(input);
    }

    @Override
    public boolean conditionSequence(List<List<Writable>> list) {
        throw new UnsupportedOperationException("not supported");
    }


    @Override
    public boolean conditionSequence(Object input) {
        throw new UnsupportedOperationException("not supported");
    }


}
