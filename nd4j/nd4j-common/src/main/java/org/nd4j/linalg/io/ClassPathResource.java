package org.nd4j.linalg.io;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;
import org.nd4j.base.Preconditions;

import java.io.*;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.attribute.FileAttribute;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

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

    /**
     * Extract the directory recursively to the specified location. Current ClassPathResource must point to
     * a directory.<br>
     * For example, if classpathresource points to "some/dir/", then the contents - not including the parent directory "dir" -
     * will be extracted or copied to the specified destination.<br>
     * @param destination Destination directory. Must exist
     */
    public void copyDirectory(File destination) throws IOException {
        Preconditions.checkState(destination.exists() && destination.isDirectory(), "Destination directory must exist and be a directory: %s", destination);


        URL url = this.getUrl();

        if (isJarURL(url)) {
            /*
                This is actually request for file, that's packed into jar. Probably the current one, but that doesn't matters.
             */
            InputStream stream = null;
            ZipFile zipFile = null;
            try {
                GetStreamFromZip getStreamFromZip = new GetStreamFromZip(url, path).invoke();
                ZipEntry entry = getStreamFromZip.getEntry();
                stream = getStreamFromZip.getStream();
                zipFile = getStreamFromZip.getZipFile();

                Preconditions.checkState(entry.isDirectory(), "Source must be a directory");

                Enumeration<? extends ZipEntry> entries = zipFile.entries();
                while(entries.hasMoreElements()){
                    ZipEntry e = entries.nextElement();
                    String name = e.getName();
                    if(name.startsWith(this.path)){
                        if(name.equalsIgnoreCase(this.path)){
                            //Skip the root dir
                            continue;
                        }

                        String relativePath = name.substring(this.path.length());

                        File extractTo = new File(destination, relativePath);
                        if(e.isDirectory()){
                            extractTo.mkdirs();
                        } else {
                            try(BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(extractTo))){
                                InputStream is = getInputStream(name, clazz, classLoader);
                                IOUtils.copy(is, bos);
                            }
                        }
                    }
                }

                stream.close();
                zipFile.close();
            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                if(stream != null)
                    IOUtils.closeQuietly(stream);
                if(zipFile != null)
                    IOUtils.closeQuietly(zipFile);
            }

        } else {
            File source = new File(path);
            Preconditions.checkState(source.isDirectory(), "Source must be a directory: %s", source);
            Preconditions.checkState(destination.exists() && destination.isDirectory(), "Destination must be a directory and must exist: %s", destination);
            FileUtils.copyDirectory(source, destination);
        }
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
        return getInputStream(path, clazz, classLoader);
    }


    private static InputStream getInputStream(String path, Class<?> clazz, ClassLoader classLoader) throws IOException {
        InputStream is;
        if (clazz != null) {
            is = clazz.getResourceAsStream(path);
        } else {
            is = classLoader.getResourceAsStream(path);
        }

        if (is == null) {
            throw new FileNotFoundException(path + " cannot be opened because it does not exist");
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

    /**
     *  Returns URL of the requested resource
     *
     * @return URL of the resource, if it's available in current Jar
     */
    private URL getUrl() {
        ClassLoader loader = null;
        try {
            loader = Thread.currentThread().getContextClassLoader();
        } catch (Exception e) {
            // do nothing
        }

        if (loader == null) {
            loader = ClassPathResource.class.getClassLoader();
        }

        URL url = loader.getResource(this.path);
        if (url == null) {
            // try to check for mis-used starting slash
            // TODO: see TODO below
            if (this.path.startsWith("/")) {
                url = loader.getResource(this.path.replaceFirst("[\\\\/]", ""));
                if (url != null)
                    return url;
            } else {
                // try to add slash, to make clear it's not an issue
                // TODO: change this mechanic to actual path purifier
                url = loader.getResource("/" + this.path);
                if (url != null)
                    return url;
            }
            throw new IllegalStateException("Resource '" + this.path + "' cannot be found.");
        }
        return url;
    }

    /**
     * Checks, if proposed URL is packed into archive.
     *
     * @param url URL to be checked
     * @return True, if URL is archive entry, False otherwise
     */
    private static boolean isJarURL(URL url) {
        String protocol = url.getProtocol();
        return "jar".equals(protocol) || "zip".equals(protocol) || "wsjar".equals(protocol)
                || "code-source".equals(protocol) && url.getPath().contains("!/");
    }

    private class GetStreamFromZip {
        private URL url;
        private ZipFile zipFile;
        private ZipEntry entry;
        private InputStream stream;
        private String resourceName;

        public GetStreamFromZip(URL url, String resourceName) {
            this.url = url;
            this.resourceName = resourceName;
        }

        public URL getUrl() {
            return url;
        }

        public ZipFile getZipFile() {
            return zipFile;
        }

        public ZipEntry getEntry() {
            return entry;
        }

        public InputStream getStream() {
            return stream;
        }

        public GetStreamFromZip invoke() throws IOException {
            url = extractActualUrl(url);

            zipFile = new ZipFile(url.getFile());
            entry = zipFile.getEntry(this.resourceName);
            if (entry == null) {
                if (this.resourceName.startsWith("/")) {
                    entry = zipFile.getEntry(this.resourceName.replaceFirst("/", ""));
                    if (entry == null) {
                        throw new FileNotFoundException("Resource " + this.resourceName + " not found");
                    }
                } else
                    throw new FileNotFoundException("Resource " + this.resourceName + " not found");
            }

            stream = zipFile.getInputStream(entry);
            return this;
        }
    }

    /**
     * Extracts parent Jar URL from original ClassPath entry URL.
     *
     * @param jarUrl Original URL of the resource
     * @return URL of the Jar file, containing requested resource
     * @throws MalformedURLException
     */
    private URL extractActualUrl(URL jarUrl) throws MalformedURLException {
        String urlFile = jarUrl.getFile();
        int separatorIndex = urlFile.indexOf("!/");
        if (separatorIndex != -1) {
            String jarFile = urlFile.substring(0, separatorIndex);

            try {
                return new URL(jarFile);
            } catch (MalformedURLException var5) {
                if (!jarFile.startsWith("/")) {
                    jarFile = "/" + jarFile;
                }

                return new URL("file:" + jarFile);
            }
        } else {
            return jarUrl;
        }
    }


}
