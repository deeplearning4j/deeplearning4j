package org.deeplearning4j.arbiter.optimize.api;

import java.util.HashMap;
import java.util.Map;

public class TaskCreatorProvider {

    private static Map<Class<? extends ParameterSpace>, Class<? extends TaskCreator>> map = new HashMap<>();

    public synchronized static TaskCreator defaultTaskCreatorFor(Class<? extends ParameterSpace> paramSpaceClass){
        Class<? extends TaskCreator> c = map.get(paramSpaceClass);
        try {
            if(c == null){
                return null;
            }
            return c.newInstance();
        } catch (Exception e){
            throw new RuntimeException("Could not create new instance of task creator class: " + c, e);
        }
    }

    public synchronized static void registerDefaultTaskCreatorClass(Class<? extends ParameterSpace> spaceClass,
                                                                    Class<? extends TaskCreator> creatorClass){
        map.put(spaceClass, creatorClass);
    }

}
