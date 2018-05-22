package org.nd4j.linalg.io;

import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;

import java.io.*;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.attribute.FileAttribute;

/**
 * A slightly upgraded version of spring's
 * classpath resource
 *
 *
 */
public class ClassPathResource extends AbstractFileResolvingResource {

    private final String path;
    private ClassLoader classLoader;
    private Class<?> clazz;

    public ClassPathResource(String path) {
        this(path, (ClassLoader) null);
    }

    public ClassPathResource(String path, ClassLoader classLoader) {
        Assert.notNull(path, "Path must not be null");
        String pathToUse = StringUtils.cleanPath(path);
        if (pathToUse.startsWith("/")) {
            pathToUse = pathToUse.substring(1);
        }

        this.path = pathToUse;
        this.classLoader = classLoader != null ? classLoader : ClassUtils.getDefaultClassLoader();
    }

    public ClassPathResource(String path, Class<?> clazz) {
        Assert.notNull(path, "Path must not be null");
        this.path = StringUtils.cleanPath(path);
        this.clazz = clazz;
    }

    protected ClassPathResource(String path, ClassLoader classLoader, Class<?> clazz) {
        this.path = StringUtils.cleanPath(path);
        this.classLoader = classLoader;
        this.clazz = clazz;
    }

    public final String getPath() {
        return this.path;
    }

    public final ClassLoader getClassLoader() {
        return this.classLoader != null ? this.classLoader : this.clazz.getClassLoader();
    }

    /**
     * Get the File.
     * If the file cannot be accessed directly (for example, it is in a JAR file), we will attempt to extract it from
     * the JAR and copy it to the temporary directory, using {@link #getTempFileFromArchive()}
     *
     * @return The File, or a temporary copy if it can not be accessed directly
     * @throws IOException
     */
    @Override
    public File getFile() throws IOException {
        try{
            return super.getFile();
        } catch (FileNotFoundException e){
            //java.io.FileNotFoundException: class path resource [iris.txt] cannot be resolved to absolute file path because
            // it does not reside in the file system: jar:file:/.../dl4j-test-resources-0.9.2-SNAPSHOT.jar!/iris.txt
            return getTempFileFromArchive();
        }
    }


    /**
     * Get a temp file from the classpath.<br>
     * This is for resources where a file is needed and the classpath resource is in a jar file. The file is copied
     * to the default temporary directory, using {@link Files#createTempFile(String, String, FileAttribute[])}.
     * Consequently, the extracted file will have a different filename to the extracted one.
     *
     * @return the temp file
     * @throws IOException If an error occurs when files are being copied
     * @see #getTempFileFromArchive(File)
     */
    public File getTempFileFromArchive() throws IOException {
        return getTempFileFromArchive(null);
    }

    /**
     * Get a temp file from the classpath, and (optionally) place it in the specified directory<br>
     * Note that:<br>
     * - If the directory is not specified, the file is copied to the default temporary directory, using
     * {@link Files#createTempFile(String, String, FileAttribute[])}. Consequently, the extracted file will have a
     * different filename to the extracted one.<br>
     * - If the directory *is* specified, the file is copied directly - and the original filename is maintained
     *
     * @param rootDirectory May be null. If non-null, copy to the specified directory
     * @return the temp file
     * @throws IOException If an error occurs when files are being copied
     * @see #getTempFileFromArchive(File)
     */
    public File getTempFileFromArchive(File rootDirectory) throws IOException {
        InputStream is = getInputStream();
        File tmpFile;
        if(rootDirectory != null){
            //Maintain original file names, as it's going in a directory...
            tmpFile = new File(rootDirectory, FilenameUtils.getName(path));
        } else {
            tmpFile = Files.createTempFile(FilenameUtils.getName(path), "tmp").toFile();
        }

        tmpFile.deleteOnExit();

        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(tmpFile));

        IOUtils.copy(is, bos);
        bos.flush();
        bos.close();
        return tmpFile;

    }

    public boolean exists() {
        URL url;
        if (this.clazz != null) {
            url = this.clazz.getResource(this.path);
        } else {
            url = this.classLoader.getResource(this.path);
        }

        return url != null;
    }

    public InputStream getInputStream() throws IOException {
        InputStream is;
        if (this.clazz != null) {
            is = this.clazz.getResourceAsStream(this.path);
        } else {
            is = this.classLoader.getResourceAsStream(this.path);
        }

        if (is == null) {
            throw new FileNotFoundException(this.getDescription() + " cannot be opened because it does not exist");
        } else {
            return is;
        }
    }

    public URL getURL() throws IOException {
        URL url;
        if (this.clazz != null) {
            url = this.clazz.getResource(this.path);
        } else {
            url = this.classLoader.getResource(this.path);
        }

        if (url == null) {
            throw new FileNotFoundException(
                            this.getDescription() + " cannot be resolved to URL because it does not exist");
        } else {
            return url;
        }
    }

    public Resource createRelative(String relativePath) {
        String pathToUse = StringUtils.applyRelativePath(this.path, relativePath);
        return new ClassPathResource(pathToUse, this.classLoader, this.clazz);
    }

    public String getFilename() {
        return StringUtils.getFilename(this.path);
    }

    public String getDescription() {
        StringBuilder builder = new StringBuilder("class path resource [");
        String pathToUse = this.path;
        if (this.clazz != null && !pathToUse.startsWith("/")) {
            builder.append(ClassUtils.classPackageAsResourcePath(this.clazz));
            builder.append('/');
        }

        if (pathToUse.startsWith("/")) {
            pathToUse = pathToUse.substring(1);
        }

        builder.append(pathToUse);
        builder.append(']');
        return builder.toString();
    }

    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        } else if (!(obj instanceof ClassPathResource)) {
            return false;
        } else {
            ClassPathResource otherRes = (ClassPathResource) obj;
            return this.path.equals(otherRes.path) && ObjectUtils.nullSafeEquals(this.classLoader, otherRes.classLoader)
                            && ObjectUtils.nullSafeEquals(this.clazz, otherRes.clazz);
        }
    }

    public int hashCode() {
        return this.path.hashCode();
    }
}
