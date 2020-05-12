package org.nd4j.imports.TFGraphs;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.Map;

@AllArgsConstructor
@Data
public class TestCase {
    public String modelName;
//    public String dir;
    public Map<String,String> inputs;  //Key: variable name, values: filename (.csv)
    public Map<String,String> outputs;
    public Map<String, DataType> datatypes;
}
