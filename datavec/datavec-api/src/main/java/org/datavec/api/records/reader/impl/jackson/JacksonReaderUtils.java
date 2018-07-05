package org.datavec.api.records.reader.impl.jackson;

import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.core.type.TypeReference;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class JacksonReaderUtils {

    private static final TypeReference<Map<String, Object>> typeRef = new TypeReference<Map<String, Object>>() {};

    private JacksonReaderUtils(){ }

    public static List<Writable> parseRecord(String line, FieldSelection selection, ObjectMapper mapper){
        List<Writable> out = new ArrayList<>();
        List<String[]> paths = selection.getFieldPaths();
        List<Writable> valueIfMissing = selection.getValueIfMissing();
        Map<String, Object> map;
        try {
            map = mapper.readValue(line, typeRef);
        } catch (IOException e) {
            throw new RuntimeException("Error parsing file", e);
        }

        //Now, extract out values...
        for (int i = 0; i < paths.size(); i++) {
            String[] currPath = paths.get(i);
            String value = null;
            Map<String, Object> currMap = map;
            for (int j = 0; j < currPath.length; j++) {
                if (currMap.containsKey(currPath[j])) {
                    Object o = currMap.get(currPath[j]);
                    if (j == currPath.length - 1) {
                        //Expect to get the final value
                        if (o instanceof String) {
                            value = (String) o;
                        } else if (o instanceof Number) {
                            value = o.toString();
                        } else {
                            throw new IllegalStateException(
                                    "Expected to find String on path " + Arrays.toString(currPath) + ", found "
                                            + o.getClass() + " with value " + o);
                        }
                    } else {
                        //Expect to get a map...
                        if (o instanceof Map) {
                            currMap = (Map<String, Object>) o;
                        }
                    }
                } else {
                    //Not found
                    value = null;
                    break;
                }
            }

            Writable outputWritable;
            if (value == null) {
                outputWritable = valueIfMissing.get(i);
            } else {
                outputWritable = new Text(value);
            }
            out.add(outputWritable);
        }

        return out;
    }

}
