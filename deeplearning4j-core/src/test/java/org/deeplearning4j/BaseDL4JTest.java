package org.deeplearning4j;

import org.junit.After;
import org.junit.Before;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;

public class BaseDL4JTest {

    public OpExecutioner.ProfilingMode getProfilingMode(){
        return OpExecutioner.ProfilingMode.SCOPE_PANIC;
    }

    @Before
    public void beforeTest(){
        Nd4j.getExecutioner().setProfilingMode(getProfilingMode());
    }

    @After
    public void afterTest(){
        //Attempt to keep workspaces isolated between tests
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }

}
