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

import org.nd4j.interceptor.InterceptorEnvironment;
import org.nd4j.interceptor.parser.SourceCodeIndexer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.sql.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class InterceptorPersistence {

    public static final Map<Long, StackTraceElement[]> arrCreationTraces = new ConcurrentHashMap<>();


    public static StackTraceElement[] getCreationTrace(long id) {
        return arrCreationTraces.get(id);
    }

    public static StackTraceElement[] getCreationTrace(INDArray arr) {
        return getCreationTrace(arr.getId());
    }

    public static void finishCurrentBackwardPass() {

    }

    public static void finishCurrentForwardPass() {

    }




    public static void addOpLog(OpLogEvent logEvent) {
        insertIntoDatabase(InterceptorEnvironment.CURRENT_FILE_PATH, logEvent);
    }




    public static void addToBackwardPass(INDArray...arrs) {

    }

    public static void addToForwardPass(INDArray...arrs) {

    }

    public static void addToForwardPass(INDArray arr) {
        addToForwardPass(new INDArray[]{arr});
    }

    public static void addToBackwardPass(INDArray arr) {
        addToBackwardPass(new INDArray[]{arr});
    }

    public static void bootstrapDatabase(String filePath) throws SQLException {
        System.out.println("Bootstrapping database");
        String jdbcUrl = "jdbc:h2:file:" + filePath;
        createDbUser(filePath);

        try(Connection conn = DriverManager.getConnection(jdbcUrl, InterceptorEnvironment.USER, InterceptorEnvironment.PASSWORD)) {
            createOpLogEventTable(conn);
            conn.commit();
        }
    }

    public static void createDbUser(String filePath) throws SQLException {
        String jdbcUrl = "jdbc:h2:file:" + filePath;
        Connection conn = DriverManager.getConnection(jdbcUrl, "SA", "");
        try {
            Statement stmt = conn.createStatement();
            //user sql: create user if not exists scott password 'tiger' admin;
            stmt.execute("create user if not exists nd4j password 'nd4j' admin");
        } finally {
            conn.commit();
            conn.close();
        }
    }


    public static void createOpLogEventTable(Connection conn) throws SQLException {
        try (Statement stmt = conn.createStatement()) {
            // Drop OpLogEvent table if it exists
            String dropTableSql = "DROP TABLE IF EXISTS OpLogEvent";
            stmt.execute(dropTableSql);

            // Create new OpLogEvent table
            String createTableSql = "CREATE TABLE OpLogEvent ("
                    + "id bigint auto_increment, "
                    + "opName VARCHAR(255), "
                    + "inputs LONGVARCHAR ARRAY, " // inputs are stored as an array
                    + "outputs LONGVARCHAR ARRAY, " // outputs are stored as an array
                    + "stackTrace LONGVARCHAR," // stackTrace is stored as a string
                    + "sourceCodeLine LONGVARCHAR" // stackTrace is stored as a string
                    + ")";
            stmt.execute(createTableSql);
            System.out.println("Created OpLogEvent table.");
        }
    }

    public static void createSourceCodeLineTable(String filePath, Connection conn) throws SQLException {
        try (Statement stmt = conn.createStatement()) {
            // Check if the SOURCE_CODE_INDEXER_PATH system property is defined
            if (InterceptorEnvironment.SOURCE_CODE_INDEXER_PATH != null) {
                System.out.println("Creating SourceCodeLine table");
                // Create new SourceCodeLine table
                String createTableQuery = "CREATE TABLE IF NOT EXISTS SourceCodeLine (" +
                        "id BIGINT AUTO_INCREMENT PRIMARY KEY," +
                        "packageName LONGVARCHAR," +
                        "className LONGVARCHAR," +
                        "lineNumber INT," +
                        "line LONGVARCHAR," +
                        "fileName LONGVARCHAR," +
                        "lastUpdated TIMESTAMP DEFAULT CURRENT_TIMESTAMP" +
                        ")";
                stmt.execute(createTableQuery);
                System.out.println("Created SourceCodeLine table.");

                // Create a SourceCodeIndexer and index the source code
                SourceCodeIndexer sourceCodeIndexer = new SourceCodeIndexer(new File(InterceptorEnvironment.SOURCE_CODE_INDEXER_PATH),filePath);

                // Persist the source code index to the OpLog
                sourceCodeIndexer.persistToOpLog(filePath);
            } else {
                System.out.println("SOURCE_CODE_INDEXER_PATH system property not defined. Skipping SourceCodeLine table creation.");
            }
        }
    }

    public static List<String> listTables(String filePath) {
        List<String> tables = new ArrayList<>();
        try {
            String jdbcUrl = "jdbc:h2:file:" + filePath;
            Connection conn = DriverManager.getConnection(jdbcUrl, InterceptorEnvironment.USER, InterceptorEnvironment.PASSWORD);
            DatabaseMetaData md = conn.getMetaData();
            ResultSet rs = md.getTables(null, null, "%", null);
            while (rs.next()) {
                tables.add(rs.getString(3));
            }
        } catch (SQLException e) {
            throw new RuntimeException("Failed to list tables", e);
        }
        return tables;
    }

    public static void insertIntoDatabase(String filePath, OpLogEvent logEvent)  {
        String jdbcUrl = "jdbc:h2:file:" + filePath;

        try (Connection conn = DriverManager.getConnection(jdbcUrl, InterceptorEnvironment.USER, InterceptorEnvironment.PASSWORD);
             PreparedStatement stmt = conn.prepareStatement("INSERT INTO OpLogEvent (opName, inputs, outputs, stackTrace,sourceCodeLine) VALUES (?, ?, ?, ?,?)")) {

            if(logEvent.firstNonExecutionCodeLine == null) {
                throw new IllegalArgumentException("Source code line should not be null.");
            }
            stmt.setString(1, logEvent.getOpName());
            stmt.setArray(2, conn.createArrayOf("VARCHAR", convertMapToArray(logEvent.getInputs())));
            stmt.setArray(3, conn.createArrayOf("VARCHAR", convertMapToArray(logEvent.getOutputs())));
            stmt.setString(4, logEvent.getStackTrace());
            stmt.setString(5,logEvent.firstNonExecutionCodeLine.trim());
            stmt.executeUpdate();
        } catch(Exception e) {
            throw new RuntimeException("Failed to insert OpLogEvent into database", e);
        }
    }

    public static String[] convertMapToArray(Map<Integer, String> map) {
        // Create a new array with the same size as the map
        String[] array = new String[map.size()];

        // Iterate over the map entries
        for (Map.Entry<Integer, String> entry : map.entrySet()) {
            // Get the key (integer) of the current entry
            int key = entry.getKey();

            // Use the key as the index in the array and assign the corresponding value
            array[key] = entry.getValue();
        }

        return array;
    }

    public static List<String> listTables() {
        List<String> tables = new ArrayList<>();
        try {
            String jdbcUrl = "jdbc:h2:file:" + InterceptorEnvironment.CURRENT_FILE_PATH;
            Connection conn = DriverManager.getConnection(jdbcUrl, InterceptorEnvironment.USER, InterceptorEnvironment.PASSWORD);
            DatabaseMetaData md = conn.getMetaData();
            ResultSet rs = md.getTables(null, null, "%", null);
            while (rs.next()) {
                tables.add(rs.getString(3));
            }
        } catch (SQLException e) {
            throw new RuntimeException("Failed to list tables", e);
        }
        return tables;
    }


    public static Map<Integer,String> convertResult(Object input) {
        Object[] inputArr = (Object[]) input;
        Map<Integer,String> ret = new LinkedHashMap<>();
        for (int i = 0; i < inputArr.length; i++) {
          ret.put(i,inputArr[i].toString());
        }
        return ret;
    }


    public static Map<String,List<OpLogEvent>> groupedByCodeSortedByEventId(List<OpLogEvent> logEvents) {
        return logEvents.stream().collect(Collectors.groupingBy(OpLogEvent::getFirstNonExecutionCodeLine, Collectors.toList()));
    }

    public static List<OpLogEvent> filterByOpName(String filePath, String opName) throws SQLException {
        List<OpLogEvent> filteredEvents = new ArrayList<>();

        String jdbcUrl = "jdbc:h2:file:" + filePath;

        try (Connection conn = DriverManager.getConnection(jdbcUrl, InterceptorEnvironment.USER, InterceptorEnvironment.PASSWORD);
             PreparedStatement stmt = conn.prepareStatement("SELECT * FROM OpLogEvent WHERE opName = ?")) {

            stmt.setString(1, opName);

            try (ResultSet rs = stmt.executeQuery()) {
                while (rs.next()) {
                    OpLogEvent event = OpLogEvent.builder()
                            .firstNonExecutionCodeLine(rs.getString("sourceCodeLine"))
                            .eventId(rs.getLong("id"))
                            .opName(rs.getString("opName"))
                            .inputs(convertResult((rs.getArray("inputs").getArray())))
                            .outputs(convertResult(rs.getArray("outputs").getArray()))
                            .stackTrace(rs.getString("stackTrace"))
                            .build();
                    filteredEvents.add(event);
                }
            }
        }

        return filteredEvents;
    }

    public static Set<String> getUniqueOpNames(String filePath) throws SQLException {
        Set<String> uniqueOpNames = new HashSet<>();
        String jdbcUrl = "jdbc:h2:file:" + filePath;

        try (Connection conn = DriverManager.getConnection(jdbcUrl, InterceptorEnvironment.USER, InterceptorEnvironment.PASSWORD);
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery("SELECT DISTINCT opName FROM OPLOGEVENT")) {

            while (rs.next()) {
                uniqueOpNames.add(rs.getString("opName"));
            }
        }

        return uniqueOpNames;
    }
}
