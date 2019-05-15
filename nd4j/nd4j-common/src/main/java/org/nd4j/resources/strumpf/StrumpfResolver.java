package org.nd4j.resources.strumpf;

import lombok.NonNull;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.resources.Resolver;

import java.io.*;
import java.util.Arrays;
import java.util.List;

/**
 * Resource resources based on Strumpf resource files, or standard files<br>
 * https://github.com/deeplearning4j/strumpf
 * <br>
 * Note that resource files (those with path ending with {@link #REF}) point to remote files, that will be downloaded,
 * decompressed and cached locally.<br>
 * The default cache location is {@link #DEFAULT_CACHE_DIR}; this can be overridden by setting the {@link #TEST_CACHE_DIR_SYSTEM_PROPERTY}
 * system property.<br>
 * <br>
 * <br>
 * Two resolution methods are supported:<br>
 * 1. Resolving from the classpath<br>
 * 2. Resolving from one of any specified directories<br>
 * <br>
 * Resolving from specified directories: You can point this to one or more local directories (rather than relying on
 * classpath) when resolving resources. This can be done by setting the {@link #LOCAL_DIRS_SYSTEM_PROPERTY}
 *
 * @author Alex Black
 */
public class StrumpfResolver implements Resolver {
    public static final String TEST_CACHE_DIR_SYSTEM_PROPERTY = "ai.skymind.test.resources.dir";
    public static final String LOCAL_DIRS_SYSTEM_PROPERTY = "ai.skymind.strumpf.resource.dirs";
    public static final String DEFAULT_CACHE_DIR = new File(System.getProperty("user.home"), ".skymind/test_resources").getAbsolutePath();
    public static final String REF = ".resource_reference";

    protected final List<String> localResourceDirs;
    protected final File cacheDir;

    public StrumpfResolver(){

        String localDirs = System.getProperty(LOCAL_DIRS_SYSTEM_PROPERTY, null);

        if(localDirs != null && !localDirs.isEmpty()) {
            String[] split = localDirs.split(",");
            localResourceDirs = Arrays.asList(split);
        } else {
            localResourceDirs = null;
        }

        String cd = System.getProperty(TEST_CACHE_DIR_SYSTEM_PROPERTY, DEFAULT_CACHE_DIR);
        cacheDir = new File(cd);
        cacheDir.mkdirs();
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

        //Second: Check classpath
        ClassPathResource cpr = new ClassPathResource(resourcePath + REF);
        if(cpr.exists()){
            return true;
        }

        cpr = new ClassPathResource(resourcePath);
        if(cpr.exists()){
            return true;
        }

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
                    return rf.localFile(cacheDir);
                }
            }
        }


        //Second: Check classpath for references (and actual file)
        ClassPathResource cpr = new ClassPathResource(resourcePath + REF);
        if(cpr.exists()){
            ResourceFile rf;
            try {
                rf = ResourceFile.fromFile(cpr.getFile());
            } catch (IOException e){
                throw new RuntimeException(e);
            }
            return rf.localFile(cacheDir);
        }

        cpr = new ClassPathResource(resourcePath);
        if(cpr.exists()){
            try {
                return cpr.getFile();
            } catch (IOException e){
                throw new RuntimeException(e);
            }
        }

        throw new RuntimeException("Could not find resource file that should exist: " + resourcePath);
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
        return cacheDir;
    }


    protected void assertExists(String resourcePath){
        if(!exists(resourcePath)){
            throw new IllegalStateException("Could not find resource with path \"" + resourcePath + "\" in local directories (" +
                    localResourceDirs + ") or in classpath");
        }
    }


}
