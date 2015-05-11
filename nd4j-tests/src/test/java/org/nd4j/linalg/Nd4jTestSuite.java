/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg;

import junit.framework.TestSuite;
import org.junit.runner.RunWith;
import org.junit.runners.AllTests;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.reflections.Reflections;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.*;

/**
 * Created by agibsoncccc on 5/11/15.
 */

@RunWith(AllTests.class)
public class Nd4jTestSuite {



    public static TestSuite suite() throws Exception  {
        TestSuite testSuite = new TestSuite();
        ServiceLoader<Nd4jBackend> backends = ServiceLoader.load(Nd4jBackend.class);
        Iterator<Nd4jBackend> backendIterator = backends.iterator();
        Reflections reflections = new Reflections("org.nd4j");
        Set<Class<? extends BaseNd4jTest>> testClasses = reflections.getSubTypesOf(BaseNd4jTest.class);
        List<Nd4jBackend> nd4jBackends = new ArrayList<>();
        while(backendIterator.hasNext()) {
            nd4jBackends.add(backendIterator.next());
        }

        for(Class<? extends BaseNd4jTest> clazz : testClasses) {
            for(Nd4jBackend backend : nd4jBackends) {
                if(backend.canRun()) {
                    Constructor<BaseNd4jTest> constructor = (Constructor<BaseNd4jTest>) clazz.getConstructor(String.class,Nd4jBackend.class);
                    Method[]  methods = clazz.getDeclaredMethods();
                    for(Method method : methods)
                        testSuite.addTest(constructor.newInstance(method.getName(),backend));

                }


            }
        }
        return testSuite;


    }

}
