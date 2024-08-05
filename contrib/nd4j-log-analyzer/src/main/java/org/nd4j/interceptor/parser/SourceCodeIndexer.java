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
package org.nd4j.interceptor.parser;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.SneakyThrows;
import org.nd4j.shade.guava.collect.HashBasedTable;
import org.nd4j.shade.guava.collect.Table;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.*;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.nd4j.interceptor.InterceptorEnvironment.PASSWORD;
import static org.nd4j.interceptor.InterceptorEnvironment.USER;

@NoArgsConstructor
@Data
@JsonSerialize(using = SourceCodeIndexerSerializer.class)
@JsonDeserialize(using = SourceCodeIndexerDeserializer.class)
public class SourceCodeIndexer {

    private Table<String,Integer, SourceCodeLine> index = HashBasedTable.create();

    public SourceCodeIndexer(File dl4jRoot,String dbPath) {
        initSourceRoot(dl4jRoot,dbPath);
    }


    public SourceCodeLine getSourceCodeLine(String fullClassName, int lineNumber) {
        return index.get(fullClassName, lineNumber);
    }

    public Set<String> getClasses() {
        return index.rowKeySet();
    }


    public void persistToOpLog(String dbPath) {
        String jdbcUrl = "jdbc:h2:file:" + dbPath + ";";
        Set<SourceCodeLine> lines = index.values().stream().collect(Collectors.toSet());
        System.out.println("Finished indexing.");
        String insertQuery = "INSERT INTO SourceCodeLine(className, lineNumber, line, packageName, fileName, lastUpdated) VALUES (?, ?, ?, ?, ?, ?)";
        String updateQuery = "UPDATE SourceCodeLine SET line = ?, lastUpdated = ? WHERE className = ? AND lineNumber = ?";

        try (Connection conn = DriverManager.getConnection(jdbcUrl, USER, PASSWORD)) {
            conn.setAutoCommit(false);

            try (PreparedStatement insertStmt = conn.prepareStatement(insertQuery);
                 PreparedStatement updateStmt = conn.prepareStatement(updateQuery)) {

                for (SourceCodeLine line : lines) {
                    // Check if the line already exists in the database
                    String selectQuery = "SELECT * FROM SourceCodeLine WHERE className = ? AND lineNumber = ?";
                    try (PreparedStatement selectStmt = conn.prepareStatement(selectQuery)) {
                        selectStmt.setString(1, line.getClassName());
                        selectStmt.setInt(2, line.getLineNumber());
                        ResultSet resultSet = selectStmt.executeQuery();

                        if (resultSet.next()) {
                            // Line already exists, check if it needs to be updated
                            String existingLine = resultSet.getString("line");
                            Timestamp existingTimestamp = resultSet.getTimestamp("lastUpdated");
                            File file = new File(line.getFileName());
                            long fileLastModified = file.lastModified();

                            if (!existingLine.equals(line.getLine()) || existingTimestamp.getTime() < fileLastModified) {
                                // Line content has changed or the file has been updated, update the line
                                updateStmt.setString(1, line.getLine());
                                updateStmt.setTimestamp(2, new Timestamp(fileLastModified));
                                updateStmt.setString(3, line.getClassName());
                                updateStmt.setInt(4, line.getLineNumber());
                                updateStmt.addBatch();
                            }
                        } else {
                            // Line doesn't exist, insert a new line
                            insertStmt.setString(1, line.getClassName());
                            insertStmt.setInt(2, line.getLineNumber());
                            insertStmt.setString(3, line.getLine());
                            insertStmt.setString(4, line.getPackageName());
                            insertStmt.setString(5, line.getFileName());
                            insertStmt.setTimestamp(6, new Timestamp(new File(line.getFileName()).lastModified()));
                            insertStmt.addBatch();
                        }
                    }
                }

                insertStmt.executeBatch();
                updateStmt.executeBatch();
                conn.commit();
            } catch (SQLException e) {
                conn.rollback();
                throw new RuntimeException("Failed to persist source code index to OpLog", e);
            }
        } catch (SQLException e) {
            throw new RuntimeException("Failed to persist source code index to OpLog", e);
        }


    }
    @SneakyThrows
    public void initSourceRoot(File nd4jApiRootDir,String dbPath) {
        CombinedTypeSolver typeSolver = new CombinedTypeSolver();
        typeSolver.add(new ReflectionTypeSolver(false));
        typeSolver.add(new JavaParserTypeSolver(nd4jApiRootDir));
        JavaSymbolSolver symbolSolver = new JavaSymbolSolver(typeSolver);
        StaticJavaParser.getConfiguration().setSymbolResolver(symbolSolver);

        String jdbcUrl = "jdbc:h2:file:" + dbPath + ";";
        String query = "SELECT * FROM SourceCodeLine WHERE fileName = ?";

        try (Connection conn = DriverManager.getConnection(jdbcUrl, USER, PASSWORD)) {
            Files.walk(nd4jApiRootDir.toPath()).parallel()
                    .map(Path::toFile)
                    .filter(file -> !file.isDirectory() && file.getName().endsWith(".java"))
                    .forEach(file -> {
                        try (PreparedStatement stmt = conn.prepareStatement(query)) {
                            stmt.setString(1, file.getAbsolutePath());
                            ResultSet resultSet = stmt.executeQuery();
                            if (resultSet.next()) {
                                Timestamp lastUpdatedTimestamp = resultSet.getTimestamp("lastUpdated");
                                long lastUpdatedTime = lastUpdatedTimestamp != null ? lastUpdatedTimestamp.getTime() : 0;
                                if (file.lastModified() <= lastUpdatedTime) {
                                    // Skip indexing this file if it hasn't been updated since the last indexing
                                    return;
                                }
                            }
                        } catch (SQLException e) {
                            throw new RuntimeException("Failed to check file timestamp in the database", e);
                        }
                        indexFile(file);
                    });
        } catch (SQLException e) {
            throw new RuntimeException("Failed to establish database connection", e);
        } catch (IOException e) {
            throw new RuntimeException("Failed to walk the directory", e);
        }
    }

    @SneakyThrows
    private void indexFile(File javaSourceFile) {
        System.out.println("Indexing file " + javaSourceFile.getName());
        // Parse the Java source file
        com.github.javaparser.ast.CompilationUnit cu = StaticJavaParser.parse(javaSourceFile);

        // Get all lines of the file
        List<String> lines = Files.readAllLines(javaSourceFile.toPath());

        // Get the package name
        String packageName = cu.getPackageDeclaration().map(pd -> pd.getNameAsString()).orElse("");

        // Iterate over each class in the file
        for (com.github.javaparser.ast.body.ClassOrInterfaceDeclaration cid : cu.findAll(com.github.javaparser.ast.body.ClassOrInterfaceDeclaration.class)) {
            // Iterate over each method in the class
            for (Statement md : cid.findAll(Statement.class)) {
                // Iterate over each line in the method
                for (int i = md.getBegin().get().line; i <= md.getEnd().get().line; i++) {
                    // Get the line of code
                    String line = lines.get(i - 1);
                    // Create a SourceCodeLine object for the line using the builder pattern
                    SourceCodeLine sourceCodeLine = SourceCodeLine.builder()
                            .line(line.stripLeading().stripTrailing())
                            .lineNumber(i)
                            .fileName(javaSourceFile.getAbsolutePath())
                            .className(cid.getNameAsString())
                            .packageName(packageName)
                            .build();

                    // Add the SourceCodeLine object to the index
                    index.put(sourceCodeLine.getClassName(), i, sourceCodeLine);
                }
            }
        }
    }


    public static void main(String...args) throws IOException {
        if(args.length < 1) {
            throw new IllegalArgumentException("Please provide the path to the deeplearning4j root directory");
        }
        File nd4jApiRootDir = new File(args[0]);
        SourceCodeIndexer sourceCodeIndexer = new SourceCodeIndexer(nd4jApiRootDir,new File("oplog.db").getAbsolutePath());
        ObjectMapper objectMapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        objectMapper.writeValue(new FileWriter("index.json"), sourceCodeIndexer);
    }



}
