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


package org.nd4j.libnd4j;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class CppTests {
    private  static String WORKING_DIR =  "tests_cpu/";
    private  static String LIBND4J_CHIP = System.getProperty("libnd4j.chip","cuda");
    private  static String LIBND4J_TEST_FILTER = System.getProperty("libnd4j.test.filter");
    private  static String LIBND4J_BUILD_DIR = System.getProperty("libnd4j.build.dir",".");
    private  static String LIBND4J_TEST_RUNNER = System.getProperty("libnd4j.test.runner");

    @Test
    public void cppTests() throws IOException, InterruptedException {
        System.out.println(LIBND4J_BUILD_DIR);
        List<String> commands = new ArrayList<>();
        commands.add("bash");
        commands.add("run_tests.sh");
        commands.add("--chip");
        commands.add(LIBND4J_CHIP);
        if(LIBND4J_TEST_FILTER != null && !LIBND4J_TEST_FILTER.isEmpty()) {
            commands.add("--test-filter");
            commands.add(LIBND4J_TEST_FILTER);
        }

        if(LIBND4J_TEST_RUNNER != null) {
            commands.add("--test-runner-prefix");
            commands.add(LIBND4J_TEST_RUNNER);
        }

        if(LIBND4J_BUILD_DIR.contains("target")) {
            LIBND4J_BUILD_DIR = LIBND4J_BUILD_DIR.replace("target","");
        }

        if(WORKING_DIR.contains("target")) {
            WORKING_DIR = WORKING_DIR.replace("target","");
        }


        ProcessBuilder processBuilder = new ProcessBuilder(commands);
        processBuilder.environment().put("CHIP",LIBND4J_CHIP);

        File dir = new File(LIBND4J_BUILD_DIR,WORKING_DIR);
        System.out.println("Using test directory: " + dir.getAbsolutePath());
        processBuilder.directory(new File(LIBND4J_BUILD_DIR,WORKING_DIR));
        Process process = processBuilder.start();

        Thread outThread = new Thread(() -> {
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            reader.lines().forEach(System.out::println);
        });
        outThread.start();

        // Wait for process to complete and for output to be printed
        int exitCode = process.waitFor();
        outThread.join();

        Assertions.assertEquals(0, exitCode, "Failed to run C++ tests");
    }
}
