package org.datavec.python;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

@javax.annotation.concurrent.NotThreadSafe
public class TestPythonSetupAndRun {
    @Test
    public void testPythonWithSetupAndRun() throws  Exception{
        String code = "def setup():" +
                "global counter;counter=0\n" +
                "def run(step):" +
                "global counter;" +
                "counter+=step;" +
                "return {\"counter\":counter}";
        PythonVariables pyInputs = new PythonVariables();
        pyInputs.addInt("step", 2);
        PythonVariables pyOutputs = new PythonVariables();
        pyOutputs.addInt("counter");
        PythonExecutioner.execWithSetupAndRun(code, pyInputs, pyOutputs);
        assertEquals((long)pyOutputs.getIntValue("counter"), 2L);
        pyInputs.addInt("step", 3);
        PythonExecutioner.execWithSetupAndRun(code, pyInputs, pyOutputs);
        assertEquals((long)pyOutputs.getIntValue("counter"), 5L);
    }
}