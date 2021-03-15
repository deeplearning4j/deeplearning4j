/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.common.resources.strumpf;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.nd4j.common.config.ND4JEnvironmentVars;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.resources.Resolver;

import java.io.*;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Slf4j
public class StrumpfResolver implements Resolver {
    public static final String DEFAULT_CACHE_DIR = new File(System.getProperty("user.home"), ".cache/nd4j/test_resources").getAbsolutePath();
    public static final String REF = ".resource_reference";

    protected final List<String> localResourceDirs;
    protected final File cacheDir;

    public StrumpfResolver() {

        String localDirs = System.getProperty(ND4JSystemProperties.RESOURCES_LOCAL_DIRS, null);

        if (localDirs != null && !localDirs.isEmpty()) {
            String[] split = localDirs.split(",");
            localResourceDirs = Arrays.asList(split);
        } else {
            localResourceDirs = null;
        }

        String cd = System.getenv(ND4JEnvironmentVars.ND4J_RESOURCES_CACHE_DIR);
        if(cd == null || cd.isEmpty()) {
            cd = System.getProperty(ND4JSystemProperties.RESOURCES_CACHE_DIR, DEFAULT_CACHE_DIR);
        }
        cacheDir = new File(cd);
        cacheDir.mkdirs();
    }

    public int priority() {
        return 100;
    }

    @Override
    public boolean exists(@NonNull String resourcePath) {
        //First: check local dirs (if any exist)
        if (localResourceDirs != null && !localResourceDirs.isEmpty()) {
            for (String s : localResourceDirs) {
                //Check for standard file:
                File f1 = new File(s, resourcePath);
                if (f1.exists() && f1.isFile()) {
                    //OK - found actual file
                    return true;
                }

                //Check for reference file:
                File f2 = new File(s, resourcePath + REF);
                if (f2.exists() && f2.isFile()) {
                    //OK - found resource reference
                    return false;
                }
            }
        }

        //Second: Check classpath
        ClassPathResource cpr = new ClassPathResource(resourcePath + REF);
        if (cpr.exists()) {
            return true;
        }

        cpr = new ClassPathResource(resourcePath);
        if (cpr.exists()) {
            return true;
        }

        return false;
    }

    @Override
    public boolean directoryExists(String dirPath) {
        //First: check local dirs (if any)
        if (localResourceDirs != null && !localResourceDirs.isEmpty()) {
            for (String s : localResourceDirs) {
                File f1 = new File(s, dirPath);
                if (f1.exists() && f1.isDirectory()) {
                    //OK - found directory
                    return true;
                }
            }
        }

        //Second: Check classpath
        ClassPathResource cpr = new ClassPathResource(dirPath);
        if (cpr.exists()) {
            return true;
        }

        return false;
    }

    @Override
    public File asFile(String resourcePath) {
        assertExists(resourcePath);

        if (localResourceDirs != null && !localResourceDirs.isEmpty()) {
            for (String s : localResourceDirs) {
                File f1 = new File(s, resourcePath);
                if (f1.exists() && f1.isFile()) {
                    //OK - found actual file
                    return f1;
                }

                //Check for reference file:
                File f2 = new File(s, resourcePath + REF);
                if (f2.exists() && f2.isFile()) {
                    //OK - found resource reference. Need to download to local cache... and/or validate what we have in cache
                    ResourceFile rf = ResourceFile.fromFile(s);
                    return rf.localFile(cacheDir);
                }
            }
        }


        //Second: Check classpath for references (and actual file)
        ClassPathResource cpr = new ClassPathResource(resourcePath + REF);
        if (cpr.exists()) {
            ResourceFile rf;
            try {
                rf = ResourceFile.fromFile(cpr.getFile());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            return rf.localFile(cacheDir);
        }

        cpr = new ClassPathResource(resourcePath);
        if (cpr.exists()) {
            try {
                return cpr.getFile();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        throw new RuntimeException("Could not find resource file that should exist: " + resourcePath);
    }

    @Override
    public InputStream asStream(String resourcePath) {
        File f = asFile(resourcePath);
        log.debug("Resolved resource " + resourcePath + " as file at absolute path " + f.getAbsolutePath());
        try {
            return new BufferedInputStream(new FileInputStream(f));
        } catch (FileNotFoundException e) {
            throw new RuntimeException("Error reading file for resource: \"" + resourcePath + "\" resolved to \"" + f + "\"");
        }
    }

    @Override
    public void copyDirectory(String dirPath, File destinationDir) {
        //First: check local resource dir
        boolean resolved = false;
        if (localResourceDirs != null && !localResourceDirs.isEmpty()) {
            for (String s : localResourceDirs) {
                File f1 = new File(s, dirPath);
                try {
                    FileUtils.copyDirectory(f1, destinationDir);
                    resolved = true;
                    break;
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        }

        //Second: Check classpath
        if (!resolved) {
            ClassPathResource cpr = new ClassPathResource(dirPath);
            if (cpr.exists()) {
                try {
                    cpr.copyDirectory(destinationDir);
                    resolved = true;
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        }

        if (!resolved) {
            throw new RuntimeException("Unable to find resource directory for path: " + dirPath);
        }

        //Finally, scan directory (recursively) and replace any resource files with actual files...
        final List<Path> toResolve = new ArrayList<>();
        try {
            Files.walkFileTree(destinationDir.toPath(), new SimpleFileVisitor<Path>() {
                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                    if (file.toString().endsWith(REF)) {
                        toResolve.add(file);
                    }
                    return FileVisitResult.CONTINUE;
                }
            });
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        if (toResolve.size() > 0) {
            for (Path p : toResolve) {
                File localFile = ResourceFile.fromFile(p.toFile()).localFile(cacheDir);
                String newPath = p.toFile().getAbsolutePath();
                newPath = newPath.substring(0, newPath.length() - REF.length());
                File destination = new File(newPath);
                try {
                    FileUtils.copyFile(localFile, destination);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                try {
                    FileUtils.forceDelete(p.toFile());
                } catch (IOException e) {
                    throw new RuntimeException("Error deleting temporary reference file", e);
                }
            }
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

    @Override
    public String normalizePath(@NonNull String path) {
        if(path.endsWith(REF)){
            return path.substring(0, path.length()-REF.length());
        }
        return path;
    }


    protected void assertExists(String resourcePath) {
        if (!exists(resourcePath)) {
            throw new IllegalStateException("Could not find resource with path \"" + resourcePath + "\" in local directories (" +
                    localResourceDirs + ") or in classpath");
        }
    }


}
