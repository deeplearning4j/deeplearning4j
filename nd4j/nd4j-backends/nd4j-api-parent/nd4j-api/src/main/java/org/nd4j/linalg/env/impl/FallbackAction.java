package org.nd4j.linalg.env.impl;

import lombok.val;
import org.nd4j.linalg.env.EnvironmentalAction;
import org.nd4j.linalg.factory.Nd4j;

public class FallbackAction implements EnvironmentalAction {
    @Override
    public String targetVariable() {
        return "ND4J_FALLBACK";
    }

    @Override
    public void process(String value) {
        val v = Boolean.valueOf(value);

        Nd4j.enableFallbackMode(v);
    }
}
