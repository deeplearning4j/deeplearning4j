package org.arbiter.deeplearning4j;

import java.lang.reflect.Method;

public class ReflectUtils {

    //TODO find better way to do this
    public static Method getMethodByName( Class<?> c, String name ){
        Method[] methods = c.getMethods();
        for( Method m : methods ){
            if(m.getName().equals(name)) return m;
        }
        throw new RuntimeException("Could not find method: " + name);
    }

}
