package org.deeplearning4j.arbiter.optimize.ui.misc;

import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;

/**
 * Created by Alex on 16/11/2016.
 */
public class YamlMapper {

    private static final ObjectMapper mapper = new ObjectMapper(new YAMLFactory());

    private YamlMapper(){}

    public static ObjectMapper getMapper(){
        return mapper;
    }

}
