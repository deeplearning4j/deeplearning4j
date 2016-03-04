package org.nd4j.linalg.api.parallel.tasks;

import org.nd4j.linalg.api.parallel.tasks.cpu.CPUTaskFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

/** Provides the default TaskFactory class name for a given Nd4jBackend */
public class TaskFactoryProvider {

    public static TaskFactory taskFactory;

    public static String getDefaultTaskFactoryForBackend(Nd4jBackend backend){

        String className = backend.getClass().getName().toLowerCase();

        if(className.contains("jcublas")){
//            throw new UnsupportedOperationException("Task factory for CUDA: not yet implemented");
            return CPUTaskFactory.class.getName();
        } else {
            return CPUTaskFactory.class.getName();
        }
    }
}
