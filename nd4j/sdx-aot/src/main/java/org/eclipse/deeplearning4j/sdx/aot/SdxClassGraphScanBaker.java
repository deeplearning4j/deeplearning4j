/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.eclipse.deeplearning4j.sdx.aot;

import io.github.classgraph.ClassGraph;
import io.github.classgraph.ScanResult;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Build-time tool: bakes the ClassGraph scan the samediff model importers need
 * into a JSON resource, because classpath scanning cannot run inside a native
 * image. {@code ClassGraphHolder} loads the baked result via the
 * {@code org.nd4j.samediff.frameworkimport.classgraph.scan.json} system property,
 * which {@link SdxNativeLibs#bootstrap()} sets in image mode (ADR 0109).
 *
 * <p>Invoked by exec-maven-plugin at process-classes in the {@code native}
 * profile, writing into {@code target/classes} so every image build carries a
 * fresh scan. The filter covers exactly what the importers query
 * ({@code AttributeMappingRule}/{@code TensorMappingRule} implementations and
 * the Pre/Post/NodePreProcessor hook classes): the framework-import packages
 * plus this repo's own packages where hook rules live.</p>
 */
public final class SdxClassGraphScanBaker {

    private SdxClassGraphScanBaker() {
    }

    public static void main(String[] args) throws Exception {
        Path out = Paths.get(args[0]);
        try (ScanResult scan = new ClassGraph()
                .disableModuleScanning()
                .enableMethodInfo()
                .enableAnnotationInfo()
                .enableClassInfo()
                .acceptPackages(
                        "org.nd4j.samediff.frameworkimport",
                        "org.eclipse.deeplearning4j")
                .scan()) {
            Files.write(out, scan.toJSON().getBytes(StandardCharsets.UTF_8));
            System.out.println("Baked ClassGraph scan: " + out + " (" + Files.size(out) + " bytes)");
        }
    }
}
