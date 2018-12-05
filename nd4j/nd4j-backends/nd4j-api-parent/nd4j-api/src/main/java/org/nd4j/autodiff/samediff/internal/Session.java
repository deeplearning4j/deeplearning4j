package org.nd4j.autodiff.samediff.internal;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;

public class Session {

    protected final SameDiff sameDiff;

    public Session(@NonNull SameDiff sameDiff){
        this.sameDiff = sameDiff;
    }

    /**
     *
     * @param variables       Name of the variables we want the arrays/activations for
     * @param outputWorkspace May be null. If null: returned arrays will be detached. If non-null: arrays will be in the specified workspace
     * @return
     */
    public Map<String,INDArray> output(List<String> variables, MemoryWorkspace outputWorkspace){

        //Basic plan: work backwards from the variables we want, based on the graph structure



        throw new UnsupportedOperationException("Not yet implemented");
    }




}
