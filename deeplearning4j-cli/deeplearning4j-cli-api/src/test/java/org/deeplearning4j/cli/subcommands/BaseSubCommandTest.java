package org.deeplearning4j.cli.subcommands;
import static org.junit.Assert.*;
import org.junit.*;

/**
 * Created by sonali on 2/11/15.
 */
public class BaseSubCommandTest {

    @org.junit.Test
    public void testSubCommand() {
        String[] cmd = {
                "--input", "testValue"
        };
        DummySubCommand dummySubCommand = new DummySubCommand(cmd);
        assertEquals("testValue",dummySubCommand.getDummyValue());
    }
}
