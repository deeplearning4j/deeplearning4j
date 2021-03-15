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
package org.nd4j.common.tests;

import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.resources.Resources;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.core.io.support.ResourcePatternResolver;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ResourceUtils {

    private ResourceUtils() {
    }

    /**
     * List all classpath resource files, optionally recursively, inside the specified path/directory
     * The path argument should be a directory.
     * Returns the path of the resources, normalized by {@link Resources#normalize(String)}
     *
     * @param path               Path in which to list all files
     * @param recursive          If true: list all files in subdirectories also. If false: only include files in the specified
     *                           directory, but not any files in subdirectories
     * @param includeDirectories If true: include any subdirectories in the returned list of files. False: Only return
     *                           files, not directories
     * @param extensions         Optional - may be null (or length 0). If null/length 0: files with any extension are returned
     *                           If non-null: only files matching one of the specified extensions are included.
     *                           Extensions can we specified with or without "." - i.e., "csv" and ".csv" are the same
     * @return List of files (and optionally directories) in the specified path
     */
    public static List<String> listClassPathFiles(String path, boolean recursive, boolean includeDirectories, String... extensions) {
        try {
            return listClassPathFilesHelper(path, recursive, includeDirectories, extensions);
        } catch (IOException e) {
            throw new RuntimeException("Error listing class path files", e);
        }
    }

    private static List<String> listClassPathFilesHelper(String path, boolean recursive, boolean includeDirectories, String... extensions) throws IOException {
        ResourcePatternResolver resolver = new PathMatchingResourcePatternResolver(new ClassPathResource(path).getClassLoader());

        StringBuilder sbPattern = new StringBuilder("classpath*:" + path);
        if (recursive) {
            sbPattern.append("/**/*");
        } else {
            sbPattern.append("/*");
        }

        //Normalize extensions so they are all like ".csv" etc - with leading "."
        String[] normExt = null;
        if (extensions != null && extensions.length > 0) {
            normExt = new String[extensions.length];
            for (int i = 0; i < extensions.length; i++) {
                if (!extensions[i].startsWith(".")) {
                    normExt[i] = "." + extensions[i];
                } else {
                    normExt[i] = extensions[i];
                }
            }
        }

        String pattern = sbPattern.toString();
        Resource[] resources = resolver.getResources(pattern);
        List<String> out = new ArrayList<>(resources.length);
        for (Resource r : resources) {
            String origPath = r.getURL().toString();
            int idx = origPath.indexOf(path);
            String relativePath = origPath.substring(idx);
            String resourcePath = Resources.normalizePath(relativePath);


            if (resourcePath.endsWith("/")) {
                if (includeDirectories) {
                    out.add(resourcePath);
                } else {
                    continue; //Skip directory
                }
            }

            if (normExt != null) {
                //Check if it matches any of the specified extensions
                boolean matches = false;
                for (String ext : normExt) {
                    if (resourcePath.endsWith(ext)) {
                        matches = true;
                        break;
                    }
                }
                if (matches) {
                    out.add(resourcePath);
                }
            } else {
                //Include all files
                out.add(resourcePath);
            }

        }
        return out;
    }
}
