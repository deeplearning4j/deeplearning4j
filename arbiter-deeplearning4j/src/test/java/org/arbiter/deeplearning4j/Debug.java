package org.arbiter.deeplearning4j;

import org.junit.Test;

import java.lang.reflect.Method;

/**
 * Created by Alex on 9/12/2015.
 */
public class Debug {

    @Test
    public void test() throws Exception {
        Method m = ReflectUtils.getMethodByName(Debug.class,"doSomething");

        Debug d = new Debug();

        m.invoke(d,true);
        m.invoke(d,Boolean.FALSE);


        Object o = Boolean.getBoolean("true");
    }

    public void doSomething(boolean b){
        System.out.println("doSomething called with value " + b);
    }

}
