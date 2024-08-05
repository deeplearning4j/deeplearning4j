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
package org.nd4j.interceptor.data;

import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.nd4j.interceptor.data.InterceptorPersistence.filterByOpName;
import static org.nd4j.interceptor.data.InterceptorPersistence.getUniqueOpNames;

public class JsonReport {


    private static final ObjectMapper objectMapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT)
            .registerModule(new JSONArraySerializer.JSONArraySerializerModule());


    public static void main(String...args) throws Exception {
        if(args.length < 1) {
            throw new IllegalArgumentException("Please provide the path to the oplog.db file");
        }
        final String CURRENT_FILE_PATH = new File(args[0]).getAbsolutePath();

        String directoryPath = "jsonReports";

        try {
            Path path = Paths.get(directoryPath);

            // Delete directory if it exists
            if (Files.exists(path)) {
                Files.walk(path)
                        .map(Path::toFile)
                        .forEach(File::delete);
            }

            // Create directory
            Files.createDirectories(path);
        } catch (IOException e) {
            throw new RuntimeException("Failed to create directory", e);
        }

        // Generate a JSON file for each unique op name
        Set<String> uniqueOpNames = getUniqueOpNames(CURRENT_FILE_PATH);
        for (String opName : uniqueOpNames) {
            List<OpLogEvent> events = filterByOpName(CURRENT_FILE_PATH, opName);
            Map<String,List<OpLogEvent>> eventsGrouped = InterceptorPersistence.groupedByCodeSortedByEventId(events);
            SourceCodeOpEvent sourceCodeOpEvent = SourceCodeOpEvent.builder()
                    .opLogEvents(eventsGrouped)
                    .build();
            System.out.println("Writing " + events.size() + " events for " + opName);
            File newFile = new File(directoryPath + "/" + opName + ".json");
            if(!newFile.exists()) {
                newFile.createNewFile();
            }
            objectMapper.writeValue(newFile, sourceCodeOpEvent);
        }

    }

}
