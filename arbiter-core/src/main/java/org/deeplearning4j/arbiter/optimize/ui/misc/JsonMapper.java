package org.deeplearning4j.arbiter.optimize.ui.misc;

import org.nd4j.shade.jackson.databind.ObjectMapper;

/**
 * Created by Alex on 16/11/2016.
 */
public class JsonMapper {

    private static final ObjectMapper mapper = new ObjectMapper();

    private JsonMapper(){}

    public static ObjectMapper getMapper(){
        return mapper;
    }

}
