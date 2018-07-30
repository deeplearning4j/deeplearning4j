package org.nd4j.linalg.env.impl;

import lombok.val;
import org.nd4j.linalg.env.EnvironmentalAction;
import org.nd4j.linalg.factory.Nd4j;

public class VerboseAction implements EnvironmentalAction {
    @Override
    public String targetVariable() {
        return "ND4J_VERBOSE";
    }

    @Override
    public void process(String value) {
        try {
            val v = Boolean.valueOf(value);

            Nd4j.getExecutioner().enableVerboseMode(v);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
