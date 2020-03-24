/* ******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.nd4j;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.imports.TFGraphs.TFGraphTestAllLibnd4j;
import org.nd4j.imports.TFGraphs.TFGraphTestAllSameDiff;
import org.nd4j.imports.TFGraphs.TFGraphTestList;
import org.nd4j.imports.TFGraphs.TFGraphTestZooModels;
import org.nd4j.imports.listeners.ImportModelDebugger;
import org.reflections.Reflections;
import org.reflections.scanners.MethodAnnotationsScanner;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;

import java.lang.reflect.Method;
import java.util.*;

import static org.junit.Assert.assertEquals;

/**
 * This class checks that all test classes (i.e., anything with one or more methods annotated with @Test)
 * extends BaseDl4jTest - either directly or indirectly.
 * Other than a small set of exceptions, all tests must extend this
 *
 * @author Alex Black
 */
@Slf4j
public class AssertTestsExtendBaseClass extends BaseND4JTest {

    @Override
    public long getTimeoutMilliseconds() {
        return 240000L;
    }

    //Set of classes that are exclusions to the rule (either run manually or have their own logging + timeouts)
    private static final Set<Class<?>> exclusions = new HashSet<>(Arrays.asList(
            TFGraphTestAllSameDiff.class,
            TFGraphTestAllLibnd4j.class,
            TFGraphTestList.class,
            TFGraphTestZooModels.class,
            ImportModelDebugger.class  //Run manually only, otherwise ignored
            ));

    @Test
    public void checkTestClasses(){

        Reflections reflections = new Reflections(new ConfigurationBuilder()
                .setUrls(ClasspathHelper.forPackage("org.nd4j"))
                .setScanners(new MethodAnnotationsScanner()));
        Set<Method> methods = reflections.getMethodsAnnotatedWith(Test.class);
        Set<Class<?>> s = new HashSet<>();
        for(Method m : methods){
            s.add(m.getDeclaringClass());
        }

        List<Class<?>> l = new ArrayList<>(s);
        l.sort(new Comparator<Class<?>>() {
            @Override
            public int compare(Class<?> aClass, Class<?> t1) {
                return aClass.getName().compareTo(t1.getName());
            }
        });

        int count = 0;
        for(Class<?> c : l){
            if(!BaseND4JTest.class.isAssignableFrom(c) && !exclusions.contains(c)){
                log.error("Test {} does not extend BaseND4JTest (directly or indirectly). All tests must extend this class for proper memory tracking and timeouts", c);
                count++;
            }
        }
        assertEquals("Number of tests not extending BaseND4JTest", 0, count);
    }
}
