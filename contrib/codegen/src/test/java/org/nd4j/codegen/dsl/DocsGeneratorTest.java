package org.nd4j.codegen.dsl;

import org.apache.commons.lang3.StringUtils;
import org.junit.jupiter.api.Test;
import org.nd4j.codegen.impl.java.DocsGenerator;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class DocsGeneratorTest {

    @Test
    public void testJDtoMDAdapter() {
        String original = "{@code %INPUT_TYPE% eye = eye(3,2)\n" +
                "                eye:\n" +
                "                [ 1, 0]\n" +
                "                [ 0, 1]\n" +
                "                [ 0, 0]}";
        String expected = "{ INDArray eye = eye(3,2)\n" +
                "                eye:\n" +
                "                [ 1, 0]\n" +
                "                [ 0, 1]\n" +
                "                [ 0, 0]}";
        DocsGenerator.JavaDocToMDAdapter adapter = new DocsGenerator.JavaDocToMDAdapter(original);
        String out = adapter.filter("@code", StringUtils.EMPTY).filter("%INPUT_TYPE%", "INDArray").toString();
        assertEquals(out, expected);
    }
}
