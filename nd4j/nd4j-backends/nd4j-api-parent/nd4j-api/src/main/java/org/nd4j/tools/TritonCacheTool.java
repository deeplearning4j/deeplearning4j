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

package org.nd4j.tools;

import org.nd4j.autodiff.samediff.execution.TritonCacheManager;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * CLI tool for managing Triton kernel cache bundles (.tkcache files).
 *
 * <h3>Usage:</h3>
 * <pre>
 * java -cp nd4j.jar org.nd4j.tools.TritonCacheTool export output.tkcache
 * java -cp nd4j.jar org.nd4j.tools.TritonCacheTool import bundle.tkcache
 * java -cp nd4j.jar org.nd4j.tools.TritonCacheTool inspect bundle.tkcache
 * </pre>
 */
public class TritonCacheTool {

    public static void main(String[] args) {
        if (args.length < 2) {
            printUsage();
            System.exit(1);
        }

        String command = args[0].toLowerCase();
        String filePath = args[1];

        try {
            switch (command) {
                case "export":
                    doExport(Paths.get(filePath));
                    break;
                case "import":
                    boolean skipArchCheck = args.length > 2 && "--skip-arch-check".equals(args[2]);
                    doImport(Paths.get(filePath), !skipArchCheck);
                    break;
                case "inspect":
                    doInspect(Paths.get(filePath));
                    break;
                default:
                    System.err.println("Unknown command: " + command);
                    printUsage();
                    System.exit(1);
            }
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    private static void doExport(Path outputPath) {
        if (!TritonCacheManager.isTritonAvailable()) {
            System.err.println("Triton is not available on this backend.");
            System.exit(1);
        }

        System.out.println("Exporting Triton kernel cache...");
        int count = TritonCacheManager.exportCache(outputPath);
        System.out.println("Exported " + count + " kernels to: " + outputPath.toAbsolutePath());
    }

    private static void doImport(Path bundlePath, boolean validateArch) {
        if (!TritonCacheManager.isTritonAvailable()) {
            System.err.println("Triton is not available on this backend.");
            System.exit(1);
        }

        System.out.println("Importing Triton kernel cache bundle...");
        int count = TritonCacheManager.importCache(bundlePath, validateArch);
        System.out.println("Imported " + count + " kernels from: " + bundlePath.toAbsolutePath());
        System.out.println("Kernels will be loaded from the override directory on next execution.");
    }

    private static void doInspect(Path bundlePath) {
        String manifest = TritonCacheManager.inspectBundle(bundlePath);
        System.out.println(manifest);
    }

    private static void printUsage() {
        System.out.println("Triton Kernel Cache Bundle Tool");
        System.out.println();
        System.out.println("Usage:");
        System.out.println("  TritonCacheTool export <output.tkcache>              Export current cache to bundle");
        System.out.println("  TritonCacheTool import <bundle.tkcache>              Import bundle to override dir");
        System.out.println("  TritonCacheTool import <bundle.tkcache> --skip-arch-check  Import without arch validation");
        System.out.println("  TritonCacheTool inspect <bundle.tkcache>             Show bundle manifest (JSON)");
    }
}
