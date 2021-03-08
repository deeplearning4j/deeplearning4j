/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.common.tests;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.reflections.Reflections;
import org.reflections.scanners.MethodAnnotationsScanner;
import org.reflections.util.ClasspathHelper;
import org.reflections.util.ConfigurationBuilder;

import java.lang.reflect.Method;
import java.util.*;

import static org.junit.Assert.assertEquals;

@Slf4j
public abstract class AbstractAssertTestsClass extends BaseND4JTest {

    protected abstract Set<Class<?>> getExclusions();

    protected abstract String getPackageName();

    protected abstract Class<?> getBaseClass();

    @Override
    public long getTimeoutMilliseconds() {
        return 240000L;
    }

    @Test
    public void checkTestClasses(){
        Reflections reflections = new Reflections(new ConfigurationBuilder()
                .setUrls(ClasspathHelper.forPackage(getPackageName()))
                .setScanners(new MethodAnnotationsScanner()));
        Set<Method> methods = reflections.getMethodsAnnotatedWith(Test.class);
        Set<Class<?>> s = new HashSet<>();
        for(Method m : methods){
            s.add(m.getDeclaringClass());
        }

        List<Class<?>> l = new ArrayList<>(s);
        Collections.sort(l, new Comparator<Class<?>>() {
            @Override
            public int compare(Class<?> aClass, Class<?> t1) {
                return aClass.getName().compareTo(t1.getName());
            }
        });

        int count = 0;
        for(Class<?> c : l){
            if(!getBaseClass().isAssignableFrom(c) && !getExclusions().contains(c)){
                log.error("Test {} does not extend {} (directly or indirectly). All tests must extend this class for proper memory tracking and timeouts",
                        c, getBaseClass());
                count++;
            }
        }
        //assertEquals("Number of tests not extending BaseND4JTest", 0, count);
    }
}
