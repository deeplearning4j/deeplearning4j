package org.nd4j.linalg.api.parallel.tasks;

/** Simple class to provide access to a TaskExecutor instance */
public class TaskExecutorProvider {

    public static TaskExecutor taskExecutor = new DefaultTaskExecutor();

    public static TaskExecutor getTaskExecutor(){
        return taskExecutor;
    }

}
