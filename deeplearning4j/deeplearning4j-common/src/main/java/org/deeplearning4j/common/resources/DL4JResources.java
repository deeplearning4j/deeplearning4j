package org.deeplearning4j.common.resources;

import lombok.NonNull;
import org.nd4j.base.Preconditions;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;

public class DL4JResources {

    public static final String DL4J_RESOURCES_DIR_PROPERTY = "org.deeplearning4j.dl4jresources.directory";
    public static final String DL4J_BASE_URL_PROPERTY = "org.deeplearning4j.dl4jresources.baseurl";
    private static final String DL4J_DEFAULT_URL = "http://blob.deeplearning4j.org/";

    private static File baseDirectory;
    private static String baseURL;

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

        property = System.getProperty(DL4J_BASE_URL_PROPERTY);
        if(property != null){
            baseURL = property;
        } else {
            baseURL = DL4J_DEFAULT_URL;
        }

    }

    public static void setBaseDownloadURL(@NonNull String baseDownloadURL){
        baseURL = baseDownloadURL;
    }

    public static String getBaseDownloadURL(){
        return baseURL;
    }

    public static URL getURL(String relativeToBase) throws MalformedURLException {
        return new URL(getURLString(relativeToBase));
    }

    public static String getURLString(String relativeToBase){
        if(relativeToBase.startsWith("/")){
            relativeToBase = relativeToBase.substring(1);
        }
        return baseURL + relativeToBase;
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
