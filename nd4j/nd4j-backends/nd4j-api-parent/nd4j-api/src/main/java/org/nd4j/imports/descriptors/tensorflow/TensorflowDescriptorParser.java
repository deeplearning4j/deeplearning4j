package org.nd4j.imports.descriptors.tensorflow;

import com.github.os72.protobuf351.TextFormat;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.io.ClassPathResource;
import org.tensorflow.framework.OpDef;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TensorflowDescriptorParser {

    /**
     * Get the op descriptors for tensorflow
     * @return the op descriptors for tensorflow
     * @throws Exception
     */
    public static Map<String,OpDef> opDescs() throws Exception {
        InputStream contents = new ClassPathResource("ops.proto").getInputStream();
        try (BufferedInputStream bis2 = new BufferedInputStream(contents); BufferedReader reader = new BufferedReader(new InputStreamReader(bis2))) {
            org.tensorflow.framework.OpList.Builder builder = org.tensorflow.framework.OpList.newBuilder();

            StringBuilder str = new StringBuilder();
            String line = null;
            while ((line = reader.readLine()) != null) {
                str.append(line);//.append("\n");
            }


            TextFormat.getParser().merge(str.toString(), builder);
            List<OpDef> list =  builder.getOpList();
            Map<String,OpDef> map = new HashMap<>();
            for(OpDef opDef : list) {
                map.put(opDef.getName(),opDef);
            }

            return map;

        } catch (Exception e2) {
            e2.printStackTrace();
        }

        throw new ND4JIllegalStateException("Unable to load tensorflow descriptors!");

    }


}
