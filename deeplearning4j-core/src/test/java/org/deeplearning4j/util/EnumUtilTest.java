package org.deeplearning4j.util;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 9/3/14.
 */
public class EnumUtilTest {

    private static final Logger log = LoggerFactory.getLogger(EnumUtil.class);

    @Test
    public void testGetEnum() {
        String val = "0";
        log.info(String.valueOf(EnumUtil.parse(val, OptimizationAlgorithm.class)));

    }



}
