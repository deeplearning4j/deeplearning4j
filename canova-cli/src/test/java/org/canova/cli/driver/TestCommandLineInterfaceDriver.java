/*
 *
 *  *
 *  *  * Copyright 2015 Skymind,Inc.
 *  *  *
 *  *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *  *    you may not use this file except in compliance with the License.
 *  *  *    You may obtain a copy of the License at
 *  *  *
 *  *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *  *
 *  *  *    Unless required by applicable law or agreed to in writing, software
 *  *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *  *    See the License for the specific language governing permissions and
 *  *  *    limitations under the License.
 *  *
 *
 */

package org.canova.cli.driver;

import static org.junit.Assert.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.canova.api.util.ClassPathResource;
import org.junit.Test;


public class TestCommandLineInterfaceDriver {

	@Test
	public void testMainCLIDriverEntryPoint() throws Exception {

		String[] args = { "vectorize", "-conf", "src/test/resources/csv/confs/unit_test_conf.txt" };

		CommandLineInterfaceDriver.main( args );

		String outputFile = "csv/data/uci_iris_sample.txt";

		ArrayList<String> vectors = new ArrayList<>();

		Map<String, Integer> labels = new HashMap<>();
		List<String> lines = FileUtils.readLines(new ClassPathResource(outputFile).getFile());
		for(String line : lines) {
			// process the line.
			if (!line.trim().isEmpty()) {
				vectors.add( line );

				String parts[] = line.split(" ");
				String key = parts[0];
				if (labels.containsKey(key)) {
					Integer count = labels.get(key);
					count++;
					labels.put(key, count);
				} else {
					labels.put(key, 1);
				}

			}
		}

		assertEquals(12, vectors.size());
		assertEquals(12, labels.size());
        File f = new File("/tmp/iris_unit_test_sample.txt");
        f.deleteOnExit();
        assertTrue(f.exists());


	}

}
