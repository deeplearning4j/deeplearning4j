package org.nd4j.linalg.env.impl;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.env.EnvironmentalAction;
import org.nd4j.linalg.factory.Nd4j;

@Slf4j
public class DebugAction implements EnvironmentalAction {
    @Override
    public String targetVariable() {
        return "ND4J_DEBUG";
    }

    @Override
    public void process(String value) {
        val v = Boolean.valueOf(value);

        Nd4j.getExecutioner().enableDebugMode(v);
    }
}
