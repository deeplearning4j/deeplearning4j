package org.deeplearning4j.util.reflections;

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
 * Custom Reflections library scanner for finding DL4J subtypes (custom layers, graph vertices, etc)
 *
 * @author Alex Black
 */
@EqualsAndHashCode
public class DL4JSubTypesScanner implements org.reflections.scanners.Scanner {

    private final List<String> interfaceNames;
    private final List<String> classNames;

    private Configuration configuration;
    private Multimap<String, String> store;

    public DL4JSubTypesScanner(List<Class<?>> interfaces, List<Class<?>> classes){
        interfaceNames = new ArrayList<>(interfaces.size());
        for(Class<?> c : interfaces){
            interfaceNames.add(c.getName());
        }

        classNames = new ArrayList<>(classes.size());
        for(Class<?> c : interfaces){
            classNames.add(c.getName());
        }
    }

    public void scan(Object cls) {
        String className = configuration.getMetadataAdapter().getClassName(cls);
        String superclass = configuration.getMetadataAdapter().getSuperclassName(cls);

        //Unfortunately: can't simply check if superclass is one of the classes we want
        // as this doesn't take into account the class heirarchy properly
        if(!"java.lang.Object".equals(superclass) ){
            getStore().put(superclass, className);
        }

        for (String interfaceName : (List<String>) configuration.getMetadataAdapter().getInterfacesNames(cls)) {
            if(interfaceNames.contains(interfaceName)){
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
