package org.nd4j.resources.remote;

import lombok.NonNull;
import org.nd4j.resources.Resolver;
import org.nd4j.resources.strumpf.ResourceFile;

import java.io.*;
import java.util.Arrays;
import java.util.List;

/**
 * TODO: Need some way of
 *
 */
public class RemoteResolver implements Resolver {
    public static final String LOCAL_DIRS_SYSTEM_PROPERTY = "ai.skymind.strumpf.resource.dirs";
    public static final String REF = ".resource_reference";

    protected final List<String> localResourceDirs;

    public RemoteResolver(){

        String localDirs = System.getProperty(LOCAL_DIRS_SYSTEM_PROPERTY, null);

        if(localDirs != null) {
            String[] split = localDirs.split(",");
            localResourceDirs = Arrays.asList(split);
        } else {
            localResourceDirs = null;
        }


    }

    public int priority() {
        return 100;
    }

    @Override
    public boolean exists(@NonNull String resourcePath) {
        //First: check local dirs (if any exist)
        if(localResourceDirs != null && !localResourceDirs.isEmpty()){
            for(String s : localResourceDirs){
                //Check for standard file:
                File f1 = new File(s, resourcePath);
                if(f1.exists() && f1.isFile()){
                    //OK - found actual file
                    return true;
                }

                //Check for reference file:
                File f2 = new File(s, resourcePath + REF);
                if(f2.exists() && f2.isFile()){
                    //OK - found resource reference
                    return false;
                }
            }
        }


        //Second: Check classpath (TODO)

        return false;
    }

    @Override
    public File asFile(String resourcePath) {
        assertExists(resourcePath);

        if(localResourceDirs != null && !localResourceDirs.isEmpty()){
            for(String s : localResourceDirs){
                //Check for standard file:
                File f1 = new File(s, resourcePath);
                if(f1.exists() && f1.isFile()){
                    //OK - found actual file
                    return f1;
                }

                //Check for reference file:
                File f2 = new File(s, resourcePath + REF);
                if(f2.exists() && f2.isFile()){
                    //OK - found resource reference. Need to download to local cache... and/or validate what we have in cache
                    ResourceFile rf = ResourceFile.fromFile(s);

                }
            }
        }


        //Second: Check classpath (TODO)

        return null;
    }

    @Override
    public InputStream asStream(String resourcePath) {

        File f = asFile(resourcePath);
        try {
            return new BufferedInputStream(new FileInputStream(f));
        } catch (FileNotFoundException e){
            throw new RuntimeException("Error reading file for resource: \"" + resourcePath + "\" resolved to \"" + f + "\"");
        }
    }

    @Override
    public boolean hasLocalCache() {
        return true;
    }

    @Override
    public File localCacheRoot() {
        //TODO
        return null;
    }


    protected void assertExists(String resourcePath){
        if(!exists(resourcePath)){
            throw new IllegalStateException("Could not find resource with path \"" + resourcePath + "\" in local directories (" +
                    localResourceDirs + ") or in classpath");
        }
    }


}
