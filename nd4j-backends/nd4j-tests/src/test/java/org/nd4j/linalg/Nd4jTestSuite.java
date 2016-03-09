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
    public final static String BACKENDS_TO_LOAD = "org.nd4j.linalg.tests.backendstorun";
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



}
