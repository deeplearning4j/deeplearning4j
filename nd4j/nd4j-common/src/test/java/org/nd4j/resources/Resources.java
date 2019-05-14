package org.nd4j.resources;

import lombok.NonNull;

import java.io.File;
import java.io.InputStream;
import java.util.*;

public class Resources {
    private static Resources INSTANCE = new Resources();

    protected final List<Resolver> resolvers;

    protected Resources(){

        ServiceLoader<Resolver> loader = ServiceLoader.load(Resolver.class);
        Iterator<Resolver> iter = loader.iterator();

        resolvers = new ArrayList<>();
        while(iter.hasNext()){
            Resolver r = iter.next();
            resolvers.add(r);
        }

        //Sort resolvers by priority: check resolvers with lower numbers first
        Collections.sort(resolvers, new Comparator<Resolver>() {
            @Override
            public int compare(Resolver r1, Resolver r2) {
                return Integer.compare(r1.priority(), r2.priority());
            }
        });


    }


    public static boolean exists(@NonNull String resourcePath){
        return INSTANCE.resourceExists(resourcePath);
    }

    public static File asFile(@NonNull String resourcePath){
        return INSTANCE.getAsFile(resourcePath);
    }

    public static InputStream asStream(@NonNull String resourcePath){
        return INSTANCE.getAsStream(resourcePath);
    }

    protected void checkResolvers(){
        if(resolvers.isEmpty()){
           throw new IllegalStateException("Cannot resolve resources: no Resolver instances are available." +
                   "No Strumpf backends on classpath or issue with ServiceLoader?");
        }
    }

    protected String listResolvers(){
        return "";
    }

    protected boolean resourceExists(String resourcePath){
        checkResolvers();

        for(Resolver r : resolvers){
            if(r.exists(resourcePath))
                return true;
        }

        return false;
    }

    protected File getAsFile(String resourcePath){
        checkResolvers();

        for(Resolver r : resolvers){
            if(r.exists(resourcePath)){
                return r.asFile(resourcePath);
            }
        }

        throw new IllegalStateException("Cannot resolve resource: none of " + resolvers.size() +
                " resolvers can resolve resource \"" + resourcePath + "\" - available resolvers: " + resolvers.toString());
    }

    public InputStream getAsStream(String resourcePath){
        checkResolvers();

        checkResolvers();

        for(Resolver r : resolvers){
            if(r.exists(resourcePath)){
                return r.asStream(resourcePath);
            }
        }

        throw new IllegalStateException("Cannot resolve resource: none of " + resolvers.size() +
                " resolvers can resolve resource \"" + resourcePath + "\" - available resolvers: " + resolvers.toString());
    }


}
