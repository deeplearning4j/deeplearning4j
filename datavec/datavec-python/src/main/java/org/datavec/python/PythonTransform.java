package org.datavec.python;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import javax.annotation.Nullable;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutionException;

public class PythonTransform {
    private String setupCode;
    private String execCode;
    private String code;
    private String name;
    private PythonVariables pyInputs;
    private PythonVariables pyOutputs;

    private void parseSetupAndExecCode() throws Exception{
        String startTag = "#<SETUP>";
        String endTag = "#</SETUP>";
        if (code.contains(startTag) && code.contains(endTag)){
            String[] sp1 = code.split(startTag);
            if (sp1.length > 2){
                throw new Exception("Only 1 <SETUP> tag allowed.");
            }
            String sp2[] = sp1[1].split(endTag);
            if (sp2.length > 2){
                throw new Exception("Only 1 </SETUP> tag allowed.");
            }
            setupCode = sp2[0];
            execCode = sp2[1];
        }
        else{
            execCode = code;
            setupCode = null;
        }
    }
    public PythonTransform(String name, String code) throws Exception{
        this.name = name;
        this.code = code;
        parseSetupAndExecCode();
    }
    public PythonTransform(String name, String[] code) throws Exception{
        String x = "";
        for (String line: code){
            x += line + "\n";
        }
        this.name = name;
        this.code = x;
        parseSetupAndExecCode();
    }
    public PythonTransform(String name, List<String> code) throws Exception{
        String x = "";
        for (String line: code){
            x += line + "\n";
        }
        this.name = name;
        this.code = x;
        parseSetupAndExecCode();
    }

    public PythonTransform(String name, String code, @Nullable PythonVariables pyInputs, @Nullable PythonVariables pyOutputs)throws Exception{
        this.name = name;
        this.code = code;
        this.pyInputs = pyInputs;
        this.pyOutputs = pyOutputs;
        parseSetupAndExecCode();
    }
    public PythonTransform(String name, String[] code, @Nullable PythonVariables pyInputs, @Nullable PythonVariables pyOutputs)throws Exception{
        String x = "";
        for (String line: code){
            x += line + "\n";
        }
        this.name = name;
        this.code = x;
        this.pyInputs = pyInputs;
        this.pyOutputs = pyOutputs;
        parseSetupAndExecCode();
    }
    public PythonTransform(String name, List<String> code, @Nullable PythonVariables pyInputs, @Nullable PythonVariables pyOutputs)throws Exception{
        String x = "";
        for (String line: code){
            x += line + "\n";
        }
        this.name = name;
        this.code = x;
        this.pyInputs = pyInputs;
        this.pyOutputs = pyOutputs;
        parseSetupAndExecCode();
    }

    public String getName(){
        return name;
    }

    public String getExecCode(){
        return execCode;
    }

    public String getSetupCode(){
        return setupCode;
    }
    public String getCode(){
        return code;
    }

    public PythonVariables getInputs() {
        return pyInputs;
    }

    public PythonVariables getOutputs() {
        return pyOutputs;
    }

    public static PythonTransform load(String filePath) throws Exception{
        JSONParser parser = new JSONParser();
        JSONObject jsonObject = (JSONObject) parser.parse(new FileReader(filePath));
        String code = (String)jsonObject.get("code");
        String name = (String)jsonObject.get("name");
        PythonVariables pyInputs;
        JSONArray inputsArr = (JSONArray) jsonObject.get("inputs");
        if (inputsArr == null){
            pyInputs = null;
        }
        else{
            pyInputs = PythonVariables.fromJSON(inputsArr);
        }
        PythonVariables pyOutputs;
        JSONArray outputsArr = (JSONArray) jsonObject.get("outputs");
        if (outputsArr == null){
            pyOutputs = null;
        }
        else{
            pyOutputs = PythonVariables.fromJSON(outputsArr);
        }
        return new PythonTransform(name, code, pyInputs, pyOutputs);
    }

    public void save(String filePath) throws IOException{
        JSONObject jsonObject = new JSONObject();
        jsonObject.put("code", code);
        jsonObject.put("name", name);
        JSONArray inputs;
        if (pyInputs == null){
            inputs = null;
        }
        else{
            inputs = pyInputs.toJSON();
        }
        jsonObject.put("inputs", inputs);
        JSONArray outputs;
        if (pyOutputs == null){
            outputs = null;
        }
        else{
            outputs = pyOutputs.toJSON();
        }
        jsonObject.put("outputs", outputs);
        String jsonString = jsonObject.toJSONString();
        FileWriter fw = new FileWriter(filePath);
        fw.write(jsonString);
        fw.close();
    }
}
