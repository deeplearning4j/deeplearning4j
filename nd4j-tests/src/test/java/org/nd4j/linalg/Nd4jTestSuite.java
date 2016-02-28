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

import org.junit.Ignore;
import org.junit.runners.BlockJUnit4ClassRunner;
import org.junit.runners.model.FrameworkMethod;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.reflections.Reflections;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.annotation.Annotation;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.*;
import java.util.List;

/**
 * Test suite for nd4j.
 *
 * This will run every combination of every unit test provided
 * that the backend's ordering and test line up.
 *
 * @author Adam Gibson
 */

public class Nd4jTestSuite extends BlockJUnit4ClassRunner {
    //the system property for what backends should run
    public final static String CLASSES_TO_LOAD = "org.nd4j.linalg.tests.classestorun";
    public final static String BACKENDS_TO_LOAD = "org.nd4j.linalg.tests.backendstorun";
    public final static String METHODS_TO_RUN = "org.nd4j.linalg.tests.methods";
    private static Logger log = LoggerFactory.getLogger(Nd4jTestSuite.class);
    private int curr;
    private List<FrameworkMethod> methods;
    private static List<Nd4jBackend> backends;
    static {
        ServiceLoader<Nd4jBackend> loadedBackends = ServiceLoader.load(Nd4jBackend.class);
        Iterator<Nd4jBackend> backendIterator = loadedBackends.iterator();
        backends = new ArrayList<>();
        while(backendIterator.hasNext())
            backends.add(backendIterator.next());


    }
    /**
     * Only called reflectively. Do not use programmatically.
     *
     * @param klass
     */
    public Nd4jTestSuite(Class<?> klass) throws Throwable {
        super(klass);
        methods = computeTestMethods();
    }


    /**
     * Based on the jvm arguments, an empty list is returned
     * if all backends should be run.
     * If only certain backends should run, please
     * pass a csv to the jvm as follows:
     * -Dorg.nd4j.linalg.tests.backendstorun=your.class1,your.class2
     * @return the list of backends to run
     */
    public static List<String> backendsToRun() {
        List<String> ret = new ArrayList<>();
        String val = System.getProperty(BACKENDS_TO_LOAD, "");
        if(val.isEmpty())
            return ret;

        String[] clazzes = val.split(",");

        for(String s : clazzes)
            ret.add(s);
        return ret;

    }

    /**
     * Based on the jvm arguments, an empty list is returned
     * if all backends should be run.
     * If only certain backends should run, please
     * pass a csv to the jvm as follows:
     * -Dorg.nd4j.linalg.tests.backendstorun=your.class1,your.class2
     * @return the list of backends to run
     */
    public static List<String> methodsToRun() {
        List<String> ret = new ArrayList<>();
        String val = System.getProperty(METHODS_TO_RUN, "");
        if(val.isEmpty())
            return ret;

        String[] clazzes = val.split(",");

        for(String s : clazzes)
            ret.add(s);
        return ret;

    }

    /**
     * Based on the jvm arguments, an empty list is returned
     * if all classses should be run.
     * If only certain classes should run, please
     * pass a csv to the jvm as follows:
     * -Dorg.nd4j.linalg.tests.classestorun=your.class1,your.class2
     * @return the list of backends to run
     */
    public static List<String> testClassesToRun() {
        List<String> ret = new ArrayList<>();
        String val = System.getProperty(CLASSES_TO_LOAD, "");
        if(val.isEmpty())
            return ret;

        String[] clazzes = val.split(",");

        for(String s : clazzes)
            ret.add(s);
        return ret;

    }
    /**
     * Isolated way of getting the tests to run
     * @return
     * @throws Exception
     */
    public static List<FrameworkMethod> testsToRunFramework() throws Exception {
        ServiceLoader<Nd4jBackend> backends = ServiceLoader.load(Nd4jBackend.class);
        Iterator<Nd4jBackend> backendIterator = backends.iterator();
        List<FrameworkMethod> ret = new ArrayList<>();
        //find all test classes on the class path
        Reflections reflections = new Reflections("org.nd4j");
        Set<Class<? extends BaseNd4jTest>> testClasses = reflections.getSubTypesOf(BaseNd4jTest.class);
        List<Nd4jBackend> nd4jBackends = new ArrayList<>();

        //an empty list if all backends should be run or a list of the backends to run
        //this is relative to the jvm args as described above
        List<String> classesToRun = testClassesToRun();
        List<String> backendsToRun = backendsToRun();
        List<String> methodsToRun = methodsToRun();

        while(backendIterator.hasNext()) {
            nd4jBackends.add(backendIterator.next());
        }

        for(Class<? extends BaseNd4jTest> clazz : testClasses) {
            //skip unwanted backends
            if(!classesToRun.isEmpty() && !classesToRun.contains(clazz.getName()) || Modifier.isAbstract(clazz.getModifiers()) || BaseComplexNDArrayTests.class.isAssignableFrom(clazz) || clazz.getAnnotation(Ignore.class) != null)
                continue;

            for(Nd4jBackend backend : nd4jBackends) {
                if(!backendsToRun.isEmpty() && !backendsToRun.contains(backend.getClass().getName()))
                    continue;
                Properties  backendProps = backend.getProperties();
                //only run if the hardware supports it: eg gpus
                if(backend.canRun()) {
                    //instantiate the method with the test
                    Method[]  methods = clazz.getDeclaredMethods();
                    for(Method method : methods) {
                        Annotation[] annotations = method.getDeclaredAnnotations();
                        if(annotations == null || annotations.length < 1)
                            continue;
                        if(!annotations[0].annotationType().equals(org.junit.Test.class))
                            continue;
                        if(!methodsToRun.isEmpty() && !methodsToRun.contains(method.getName()))
                            continue;
                        log.info("Constructing test for backend " + backend.getClass().getName());
                        //backout if the test ordering and backend ordering dont line up
                        //unless the ordering is a (the default) which means all
                        ret.add(new FrameworkMethod(method));

                    }

                }


            }
        }

        return ret;

    }

    /**
     * Isolated way of getting the tests to run
     * @return
     * @throws Exception
     */
    public static List<BaseNd4jTest> testsToRun() throws Exception {
        ServiceLoader<Nd4jBackend> backends = ServiceLoader.load(Nd4jBackend.class);
        Iterator<Nd4jBackend> backendIterator = backends.iterator();
        List<BaseNd4jTest> ret = new ArrayList<>();
        //find all test classes on the class path
        Reflections reflections = new Reflections("org.nd4j");
        Set<Class<? extends BaseNd4jTest>> testClasses = reflections.getSubTypesOf(BaseNd4jTest.class);
        List<Nd4jBackend> nd4jBackends = new ArrayList<>();

        //an empty list if all backends should be run or a list of the backends to run
        //this is relative to the jvm args as described above
        List<String> classesToRun = testClassesToRun();
        List<String> backendsToRun = backendsToRun();
        List<String> methodsToRun = methodsToRun();

        while(backendIterator.hasNext()) {
            nd4jBackends.add(backendIterator.next());
        }

        for(Class<? extends BaseNd4jTest> clazz : testClasses) {
            //skip unwanted backends
            if(!classesToRun.isEmpty() && !classesToRun.contains(clazz.getName()) || Modifier.isAbstract(clazz.getModifiers()) || BaseComplexNDArrayTests.class.isAssignableFrom(clazz) || clazz.getAnnotation(Ignore.class) != null)
                continue;

            for(Nd4jBackend backend : nd4jBackends) {
                if(!backendsToRun.isEmpty() && !backendsToRun.contains(backend.getClass().getName()))
                    continue;
                Properties  backendProps = backend.getProperties();
                //only run if the hardware supports it: eg gpus
                if(backend.canRun()) {
                    //instantiate the method with the test
                    Constructor<BaseNd4jTest> constructor = (Constructor<BaseNd4jTest>) clazz.getConstructor(String.class,Nd4jBackend.class);
                    Method[]  methods = clazz.getDeclaredMethods();
                    for(Method method : methods) {
                        Annotation[] annotations = method.getDeclaredAnnotations();
                        if(annotations == null || annotations.length < 1)
                            continue;
                        if(!annotations[0].annotationType().equals(org.junit.Test.class))
                            continue;
                        if(!methodsToRun.isEmpty() && !methodsToRun.contains(method.getName()))
                            continue;
                        try {
                            log.info("Constructing test for backend " + backend.getClass().getName());
                            BaseNd4jTest test = constructor.newInstance(method.getName(),backend);
                            //backout if the test ordering and backend ordering dont line up
                            //unless the ordering is a (the default) which means all
                            ret.add(test);

                        }catch(InstantiationException e) {
                            throw new RuntimeException("Failed to construct backend " + backend.getClass() + " with method " + method.getName() + " with class " + clazz.getName(),e);
                        }

                    }

                }


            }
        }

        return ret;

    }


}
