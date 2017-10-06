package org.nd4j.imports.intermediate;

import lombok.val;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class TVariableSpaceTest {
    @Before
    public void setUp() throws Exception {
    }

    @Test
    public void hasVariable() throws Exception {
        val space = new TVariableSpace();

        val variable = TVariable.builder()
                .id(-1)
                .name("somename")
                .build();

        space.addVariable(variable.getId(), variable);

        assertTrue(space.hasVariable(-1));
        assertTrue(space.hasVariable(TIndex.makeOf(-1)));
        assertTrue(space.hasVariable(TIndex.makeOf(-1, 0)));
        assertTrue(space.hasVariable("somename"));
    }

}