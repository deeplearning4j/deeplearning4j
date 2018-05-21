package org.datavec.spark.transform.transform;

import org.datavec.api.transform.TransformProcess;
import org.datavec.api.writable.Writable;
import org.datavec.spark.functions.FlatMapFunctionAdapter;

import java.util.Collections;
import java.util.List;

/**
 * Spark function for executing a transform process
 */
public class SparkTransformProcessFunctionAdapter implements FlatMapFunctionAdapter<List<Writable>, List<Writable>> {

    private final TransformProcess transformProcess;

    public SparkTransformProcessFunctionAdapter(TransformProcess transformProcess) {
        this.transformProcess = transformProcess;
    }

    @Override
    public Iterable<List<Writable>> call(List<Writable> v1) throws Exception {
        List<Writable> newList = transformProcess.execute(v1);
        if (newList == null)
            return Collections.emptyList(); //Example was filtered out
        else
            return Collections.singletonList(newList);
    }
}
