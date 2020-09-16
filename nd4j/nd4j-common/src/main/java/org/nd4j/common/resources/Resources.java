package org.nd4j.common.resources;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.resources.strumpf.StrumpfResolver;

import java.io.File;
import java.io.InputStream;
import java.util.*;

/**
 * API for accessing resources (usually test resources) from a path.
 * For example, depending on the implementation (underlying set of {@link Resolver instances} this class cas be used to
 * resolve files on the classpath, or reference files (referring to a remote file that needs to be downloaded).<br>
 * By default only the {@link StrumpfResolver} is used, but others can be added using the Java {@link ServiceLoader} mechanism
 * for {@link Resolver} class.
 *
 * @author Alex Black
 */
@Slf4j
public class Resources {
    private static Resources INSTANCE = new Resources();

    protected final List<Resolver> resolvers;

    protected Resources() {

        ServiceLoader<Resolver> loader = ServiceLoader.load(Resolver.class);
        Iterator<Resolver> iter = loader.iterator();

        resolvers = new ArrayList<>();
        resolvers.add(new StrumpfResolver());
        while (iter.hasNext()) {
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

    /**
     * Check if the specified resource exists (can be resolved by any method) hence can be loaded by {@link #asFile(String)}
     * or {@link #asStream(String)}
     *
     * @param resourcePath Path of the resource to be resolved
     * @return Whether the resource can be resolved or not
     */
    public static boolean exists(@NonNull String resourcePath) {
        return INSTANCE.resourceExists(resourcePath);
    }

    /**
     * Get the specified resource as a local file.
     * If it cannot be found (i.e., {@link #exists(String)} returns false) this method will throw an exception.
     *
     * @param resourcePath Path of the resource to get
     * @return Resource file
     */
    public static File asFile(@NonNull String resourcePath) {
        return INSTANCE.getAsFile(resourcePath);
    }

    /**
     * Get the specified resource as an input stream.<br>
     * If it cannot be found (i.e., {@link #exists(String)} returns false) this method will throw an exception.
     *
     * @param resourcePath Path of the resource to get
     * @return Resource stream
     */
    public static InputStream asStream(@NonNull String resourcePath) {
        return INSTANCE.getAsStream(resourcePath);
    }

    /**
     * Copy the contents of the specified directory (path) to the specified destination directory, resolving any resources in the process
     *
     * @param directoryPath  Directory to copy contents of
     * @param destinationDir Destination
     */
    public static void copyDirectory(@NonNull String directoryPath, @NonNull File destinationDir) {
        INSTANCE.copyDir(directoryPath, destinationDir);
    }

    /**
     * Normalize the path that may be a resource reference.
     * For example: "someDir/myFile.zip.resource_reference" --> "someDir/myFile.zip"
     * Returns null if the file cannot be resolved.
     * If the file is not a reference, the original path is returned
     */
    public static String normalizePath(String path){
        return INSTANCE.normalize(path);
    }

    protected boolean resourceExists(String resourcePath) {
        for (Resolver r : resolvers) {
            if (r.exists(resourcePath))
                return true;
        }

        return false;
    }

    protected File getAsFile(String resourcePath) {
        for (Resolver r : resolvers) {
            if (r.exists(resourcePath)) {
                return r.asFile(resourcePath);
            }
        }

        throw new IllegalStateException("Cannot resolve resource (not found): none of " + resolvers.size() +
                " resolvers can resolve resource \"" + resourcePath + "\" - available resolvers: " + resolvers.toString());
    }

    public InputStream getAsStream(String resourcePath) {
        for (Resolver r : resolvers) {
            if (r.exists(resourcePath)) {
                log.debug("Resolved resource with resolver " + r.getClass().getName() + " for path " + resourcePath);
                return r.asStream(resourcePath);
            }
        }

        throw new IllegalStateException("Cannot resolve resource (not found): none of " + resolvers.size() +
                " resolvers can resolve resource \"" + resourcePath + "\" - available resolvers: " + resolvers.toString());
    }

    public void copyDir(String directoryPath, File destinationDir) {
        for (Resolver r : resolvers) {
            if (r.directoryExists(directoryPath)) {
                r.copyDirectory(directoryPath, destinationDir);
                return;
            }
        }
    }

    public String normalize(String path){
        for(Resolver r : resolvers){
            path = r.normalizePath(path);
        }
        return path;
    }

}
