package org.nd4j.common.resources.strumpf;

import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.shade.guava.io.Files;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.codec.digest.DigestUtils;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;
import org.nd4j.common.base.Preconditions;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;

import java.io.*;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Map;

/**
 * ResourceFile - represents a Strumpf resource file - which are references to remote files with filename ending with {@link StrumpfResolver#REF}
 * @author Alex Black
 */
@AllArgsConstructor
@NoArgsConstructor
@Data
@JsonIgnoreProperties("filePath")
@Slf4j
public class ResourceFile {
    /**
     * Default value for resource downloading connection timeout - see {@link ND4JSystemProperties#RESOURCES_CONNECTION_TIMEOUT}
     */
    public static final int DEFAULT_CONNECTION_TIMEOUT = 60000;        //Timeout for connections to be established
    /**
     * Default value for resource downloading read timeout - see {@link ND4JSystemProperties#RESOURCES_READ_TIMEOUT}
     */
    public static final int DEFAULT_READ_TIMEOUT = 60000;              //Timeout for amount of time between connection established and data is available
    protected static final String PATH_KEY = "full_remote_path";
    protected static final String HASH = "_hash";
    protected static final String COMPRESSED_HASH = "_compressed_hash";

    protected static final int MAX_DOWNLOAD_ATTEMPTS = 3;

    public static final ObjectMapper MAPPER = newMapper();

    //Note: Field naming to match Strumpf JSON format
    protected int current_version;
    protected Map<String, String> v1;

    //Not in JSON:
    protected String filePath;

    public static ResourceFile fromFile(String path) {
        return fromFile(new File(path));
    }

    public static ResourceFile fromFile(File file) {
        String s;
        try {
            s = FileUtils.readFileToString(file, StandardCharsets.UTF_8);
            ResourceFile rf = MAPPER.readValue(s, ResourceFile.class);
            rf.setFilePath(file.getPath());
            return rf;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public String relativePath() {
        String hashKey = null;
        for (String key : v1.keySet()) {
            if (key.endsWith(HASH) && !key.endsWith(COMPRESSED_HASH)) {
                hashKey = key;
                break;
            }
        }
        if (hashKey == null) {
            throw new IllegalStateException("Could not find <filename>_hash in resource reference file: " + filePath);
        }

        String relativePath = hashKey.substring(0, hashKey.length() - 5); //-5 to remove "_hash" suffix
        return relativePath.replaceAll("\\\\", "/");
    }

    public boolean localFileExistsAndValid(File cacheRootDir) {

        File file = getLocalFile(cacheRootDir);
        if (!file.exists()) {
            return false;
        }

        //File exists... but is it valid?
        String sha256Property = relativePath() + HASH;
        String expSha256 = v1.get(sha256Property);

        Preconditions.checkState(expSha256 != null, "Expected JSON property %s was not found in resource reference file %s", sha256Property, filePath);

        String actualSha256 = sha256(file);
        if (!expSha256.equals(actualSha256)) {
            return false;
        }
        return true;
    }

    /**
     * Get the local file - or where it *would* be if it has been downloaded. If it does not exist, it will not be downloaded here
     *
     * @return
     */
    protected File getLocalFile(File cacheRootDir) {
        String relativePath = relativePath();

        //For resolving local files with different versions, we want paths like:
        // ".../dir/filename.txt__v1/filename.txt"
        // ".../dir/filename.txt__v2/filename.txt"
        //This is to support multiple versions of files simultaneously... for example, different projects needing different
        // versions, or supporting old versions of resource files etc

        int lastSlash = Math.max(relativePath.lastIndexOf('/'), relativePath.lastIndexOf('\\'));
        String filename;
        if (lastSlash < 0) {
            filename = relativePath;
        } else {
            filename = relativePath.substring(lastSlash + 1);
        }

        File parentDir = new File(cacheRootDir, relativePath + "__v" + current_version);
        File file = new File(parentDir, filename);
        return file;
    }

    /**
     * Get the local file - downloading and caching if required
     *
     * @return
     */
    public File localFile(File cacheRootDir) {
        if (localFileExistsAndValid(cacheRootDir)) {
            return getLocalFile(cacheRootDir);
        }

        //Need to download and extract...
        String remotePath = v1.get(PATH_KEY);
        Preconditions.checkState(remotePath != null, "No remote path was found in resource reference file %s", filePath);
        File f = getLocalFile(cacheRootDir);

        File tempDir = Files.createTempDir();
        File tempFile = new File(tempDir, FilenameUtils.getName(remotePath));

        String sha256PropertyCompressed = relativePath() + COMPRESSED_HASH;

        String sha256Compressed = v1.get(sha256PropertyCompressed);
        Preconditions.checkState(sha256Compressed != null, "Expected JSON property %s was not found in resource reference file %s", sha256PropertyCompressed, filePath);

        String sha256Property = relativePath() + HASH;
        String sha256Uncompressed = v1.get(sha256Property);

        String connTimeoutStr = System.getProperty(ND4JSystemProperties.RESOURCES_CONNECTION_TIMEOUT);
        String readTimeoutStr = System.getProperty(ND4JSystemProperties.RESOURCES_READ_TIMEOUT);
        boolean validCTimeout = connTimeoutStr != null && connTimeoutStr.matches("\\d+");
        boolean validRTimeout = readTimeoutStr != null && readTimeoutStr.matches("\\d+");

        int connectTimeout = validCTimeout ? Integer.parseInt(connTimeoutStr) : DEFAULT_CONNECTION_TIMEOUT;
        int readTimeout = validRTimeout ? Integer.parseInt(readTimeoutStr) : DEFAULT_READ_TIMEOUT;

        try {
            boolean correctHash = false;
            for (int tryCount = 0; tryCount < MAX_DOWNLOAD_ATTEMPTS; tryCount++) {
                try {
                    if (tempFile.exists())
                        tempFile.delete();
                    log.info("Downloading remote resource {} to {}", remotePath, tempFile);
                    FileUtils.copyURLToFile(new URL(remotePath), tempFile, connectTimeout, readTimeout);
                    //Now: check if downloaded archive hash is OK
                    String hash = sha256(tempFile);
                    correctHash = sha256Compressed.equals(hash);
                    if (!correctHash) {
                        log.warn("Download of file {} failed: expected hash {} vs. actual hash {}", remotePath, sha256Compressed, hash);
                        continue;
                    }
                    log.info("Downloaded {} to temporary file {}", remotePath, tempFile);
                    break;
                } catch (Throwable t) {
                    if (tryCount == MAX_DOWNLOAD_ATTEMPTS - 1) {
                        throw new RuntimeException("Error downloading test resource: " + remotePath, t);
                    }
                    log.warn("Error downloading test resource, retrying... {}", remotePath, t);
                }
            }

            if (!correctHash) {
                throw new RuntimeException("Could not successfully download with correct hash file after " + MAX_DOWNLOAD_ATTEMPTS +
                        " attempts: " + remotePath);
            }

            //Now, extract:
            f.getParentFile().mkdirs();
            try (OutputStream os = new BufferedOutputStream(new FileOutputStream(f));
                 InputStream is = new BufferedInputStream(new GzipCompressorInputStream(new FileInputStream(tempFile)))) {
                IOUtils.copy(is, os);
            } catch (IOException e) {
                throw new RuntimeException("Error extracting resource file", e);
            }
            log.info("Extracted {} to {}", tempFile, f);

            //Check extracted file hash:
            String extractedHash = sha256(f);
            if (!extractedHash.equals(sha256Uncompressed)) {
                throw new RuntimeException("Extracted file hash does not match expected hash: " + remotePath +
                        " -> " + f.getAbsolutePath() + " - expected has " + sha256Uncompressed + ", actual hash " + extractedHash);
            }

        } finally {
            tempFile.delete();
        }

        return f;
    }

    public static String sha256(File f) {
        try (InputStream is = new BufferedInputStream(new FileInputStream(f))) {
            return DigestUtils.sha256Hex(is);
        } catch (IOException e) {
            throw new RuntimeException("Error when hashing file: " + f.getPath(), e);
        }
    }


    public static final ObjectMapper newMapper() {
        ObjectMapper ret = new ObjectMapper();
        ret.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        ret.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        ret.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        ret.enable(SerializationFeature.INDENT_OUTPUT);
        return ret;
    }
}
