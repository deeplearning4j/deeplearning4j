/*-
 *  * Copyright 2016 Skymind, Inc.
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
 */

package org.datavec.api.util.reflections;

import com.google.common.base.Predicate;
import com.google.common.collect.Multimap;
import lombok.EqualsAndHashCode;
import org.reflections.Configuration;
import org.reflections.ReflectionsException;
import org.reflections.scanners.Scanner;
import org.reflections.vfs.Vfs;

import java.util.ArrayList;
import java.util.List;

/**
 * Custom Reflections library scanner for finding Datavec subtypes (custom Transforms, Conditions, etc)
 *
 * @author Alex Black
 */
@EqualsAndHashCode
public class DataVecSubTypesScanner implements Scanner {

    private final List<String> interfaceNames;
    private final List<String> classNames;

    private Configuration configuration;
    private Multimap<String, String> store;

    public DataVecSubTypesScanner(List<Class<?>> interfaces, List<Class<?>> classes) {
        interfaceNames = new ArrayList<>(interfaces.size());
        for (Class<?> c : interfaces) {
            interfaceNames.add(c.getName());
        }

        classNames = new ArrayList<>(classes.size());
        for (Class<?> c : interfaces) {
            classNames.add(c.getName());
        }
    }

    public void scan(Object cls) {
        String className = configuration.getMetadataAdapter().getClassName(cls);
        String superclass = configuration.getMetadataAdapter().getSuperclassName(cls);

        //Unfortunately: can't simply check if superclass is one of the classes we want
        // as this doesn't take into account the class heirarchy properly
        if (!"java.lang.Object".equals(superclass)) {
            getStore().put(superclass, className);
        }

        for (String interfaceName : (List<String>) configuration.getMetadataAdapter().getInterfacesNames(cls)) {
            if (interfaceNames.contains(interfaceName)) {
                getStore().put(interfaceName, className);
            }
        }

    }

    @Override
    public boolean acceptsInput(String file) {
        return configuration.getMetadataAdapter().acceptsInput(file);
    }

    @Override
    public Object scan(Vfs.File file, Object classObject) {
        if (classObject == null) {
            try {
                classObject = configuration.getMetadataAdapter().getOfCreateClassObject(file);
            } catch (Exception e) {
                throw new ReflectionsException("could not create class object from file " + file.getRelativePath());
            }
        }
        scan(classObject);
        return classObject;
    }

    @Override
    public void setConfiguration(final Configuration configuration) {
        this.configuration = configuration;
    }

    @Override
    public Multimap<String, String> getStore() {
        return store;
    }

    @Override
    public void setStore(final Multimap<String, String> store) {
        this.store = store;
    }

    @Override
    public Scanner filterResultsBy(Predicate<String> filter) {
        //NO op
        return this;
    }

    @Override
    public boolean acceptResult(final String fqn) {
        return fqn != null && (classNames.contains(fqn) || interfaceNames.contains(fqn));
    }
}
