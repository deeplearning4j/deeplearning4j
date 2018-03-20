package org.datavec.local.transforms.transform;

import org.datavec.api.transform.TransformProcess;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.functions.FlatMapFunctionAdapter;

import java.util.Collections;
import java.util.List;

/**
 * Function for executing a transform process
 */
public class LocalTransformProcessFunctionAdapter implements FlatMapFunctionAdapter<List<Writable>, List<Writable>> {

    private final TransformProcess transformProcess;

    public LocalTransformProcessFunctionAdapter(TransformProcess transformProcess) {
        this.transformProcess = transformProcess;
    }

    @Override
    public List<List<Writable>> call(List<Writable> v1) throws Exception {
        List<Writable> newList = transformProcess.execute(v1);
        if (newList == null)
            return Collections.emptyList(); //Example was filtered out
        else
            return Collections.singletonList(newList);
    }
}
