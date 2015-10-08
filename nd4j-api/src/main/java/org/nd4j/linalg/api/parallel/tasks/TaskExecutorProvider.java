package org.nd4j.linalg.api.parallel.tasks;

public class TaskExecutorProvider {

    public static TaskExecutor taskExecutor;

    static{
        taskExecutor = new DefaultTaskExecutor();
    }

    public static TaskExecutor getTaskExecutor(){
        return taskExecutor;
    }

}
