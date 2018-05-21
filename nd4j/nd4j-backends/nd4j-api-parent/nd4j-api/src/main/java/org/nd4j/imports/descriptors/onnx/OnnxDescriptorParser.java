package org.nd4j.imports.descriptors.onnx;

import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Load all of the onnx op descriptors from the classpath.
 *
 * @author Adam Gibson
 */
public class OnnxDescriptorParser {


    /**
     * Get the onnx op descriptors by name
     * @return the onnx op descriptors by name
     * @throws Exception
     */
    public static Map<String,OpDescriptor> onnxOpDescriptors() throws Exception {
        try(InputStream is = new ClassPathResource("onnxops.json").getInputStream()) {
            ObjectMapper objectMapper = new ObjectMapper();
            OnnxDescriptor opDescriptor = objectMapper.readValue(is,OnnxDescriptor.class);
            Map<String,OpDescriptor> descriptorMap = new HashMap<>();
            for(OpDescriptor descriptor : opDescriptor.getDescriptors()) {
                descriptorMap.put(descriptor.getName(),descriptor);
            }



            return descriptorMap;
        }
    }


}
