package org.deeplearning4j.common.resources;

import org.nd4j.base.Preconditions;

import java.io.File;

public class DL4JResources {

    public static String DL4J_RESOURCES_DIR_PROPERTY = "org.deeplearning4j.dl4jresources.directory";

    private static File baseDirectory;

    static {
        String property = System.getProperty(DL4J_RESOURCES_DIR_PROPERTY);
        if(property != null){
            baseDirectory = new File(property);
        } else {
            baseDirectory = new File(System.getProperty("user.home"), ".deeplearning4j");
        }

        if(!baseDirectory.exists()){
            baseDirectory.mkdirs();
        }

    }

    public static void resetDefaultDirectory(){
        baseDirectory = new File(System.getProperty("user.home"), ".deeplearning4j");
    }

    public static void setBaseDirectory(File f){
        Preconditions.checkState(f.exists() && f.isDirectory(), "Specified base directory does not exist and/or is not a directory: %s", f.getAbsolutePath());

    }

    public static File getBaseDirectory(){
        return baseDirectory;
    }

    public static File getDirectory(ResourceType resourceType, String resourceName){
        File f = new File(baseDirectory, resourceType.resourceName());
        f = new File(f, resourceName);
        f.mkdirs();
        return f;
    }




}
