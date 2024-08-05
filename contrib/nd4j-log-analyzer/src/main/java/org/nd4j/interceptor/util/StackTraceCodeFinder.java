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
package org.nd4j.interceptor.util;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Pattern;

public class StackTraceCodeFinder {

    private static final Map<String, Path> filePathCache = new HashMap<>();

    public static String getFirstLineOfCode(String rootDirectory, StackTraceElement[] stackTrace) {
        if (rootDirectory == null) {
            return null;
        }

        if(!new File(rootDirectory).exists()) {
            throw new IllegalArgumentException("Root directory does not exist. Unable to scan code path.");
        }

        Set<String> skipPatterns = new HashSet<>(Arrays.asList(
                "org\\.nd4j\\.linalg\\.api\\.ops.*",
                "org\\.nd4j\\.interceptor.*",
                "org\\.nd4j\\.linalg\\.api\\.ops\\.executioner.*",
                "java\\.lang\\.*",
                "org\\.nd4j\\.linalg\\.cpu\\.nativecpu\\.ops.*",
                "org\\.nd4j\\.linalg\\.jcublas\\.ops\\.executioner.*",
                "org\\.nd4j\\.linalg\\.factory.*",
                "org\\.nd4j\\.linalg\\.api\\.ndarray.*",
                "org\\.nd4j\\.linalg\\.api\\.blas\\.impl.*",
                "org\\.deeplearning4j\\.nn\\.updater.*"
        ));

        for (StackTraceElement element : stackTrace) {
            String className = element.getClassName();
            String packageName = extractPackageName(className);
            if (shouldSkip(packageName, skipPatterns)) {
                continue;
            }

            String line = getLineOfCode(element, rootDirectory);
            if (line != null) {
                return line;
            }
        }

       return null;
    }

    public static String extractPackageName(String fullyQualifiedClassName) {
        int lastDotIndex = fullyQualifiedClassName.lastIndexOf('.');
        if (lastDotIndex > 0) {
            return fullyQualifiedClassName.substring(0, lastDotIndex);
        }
        return ""; // Default package (no package)
    }


    public static String getLineOfCode(StackTraceElement element, String rootDirectory) {
        String className = element.getClassName();
        int lineNumber = element.getLineNumber();

        Path filePath = resolveClassFile(rootDirectory, className);

        if (filePath != null) {
            try {
                List<String> lines = Files.readAllLines(filePath);
                if (lineNumber >= 1 && lineNumber <= lines.size()) {
                    return lines.get(lineNumber - 1);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return null;
    }

    private static boolean shouldSkip(String className, Set<String> skipPatterns) {
        for (String pattern : skipPatterns) {
            if (Pattern.matches(pattern, className)) {
                return true;
            }
        }
        return false;
    }

    public static Path resolveClassFile(String rootDirectory, String fullyQualifiedName) {
        if (filePathCache.containsKey(fullyQualifiedName)) {
            return filePathCache.get(fullyQualifiedName);
        }

        String relativePath = fullyQualifiedName.replace('.', File.separatorChar) + ".java";
        List<Path> sourceRoots = findSourceRoots(rootDirectory);

        for (Path sourceRoot : sourceRoots) {
            Path filePath = sourceRoot.resolve(relativePath);
            if (Files.exists(filePath)) {
                filePathCache.put(fullyQualifiedName, filePath);
                return filePath;
            }
        }

        return null;
    }

    private static List<Path> findSourceRoots(String rootDirectory) {
        StackTraceCodeFinderFileVisitor fileVisitor = new StackTraceCodeFinderFileVisitor();
        try {
            Files.walkFileTree(Paths.get(rootDirectory), fileVisitor);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return fileVisitor.sourceRoots;
    }
}