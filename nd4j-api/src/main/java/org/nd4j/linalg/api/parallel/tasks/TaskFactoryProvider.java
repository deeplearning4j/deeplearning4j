package org.nd4j.linalg.api.parallel.tasks;

import org.nd4j.linalg.api.parallel.tasks.cpu.CPUTaskFactory;
import org.nd4j.linalg.factory.Nd4j;

public class TaskFactoryProvider {

    public static TaskFactory taskFactory;

    public static TaskFactory getTaskFactory(){
        //TODO switch to static initializer for this?

        if(taskFactory==null) {
            //Detect which backend we are on, and return an appropriate TaskFactory
            //TODO: Got to be a better way than this
            String factoryClassName = Nd4j.factory().getClass().toString().toLowerCase();
            if (factoryClassName.contains("jcublas")) {
                throw new RuntimeException("Not yet implemented");
            } else {
                //JBlas, Java, x86 etc: all CPU backends
                taskFactory = new CPUTaskFactory();
            }
        }
        return taskFactory;
    }
}
