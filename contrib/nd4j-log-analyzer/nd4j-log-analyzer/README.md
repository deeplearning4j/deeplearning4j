# ND4J Log Analyzer

## Overview

ND4J Log Analyzer is a Java agent designed to record ND4J operation executions in an H2 database and index a specified DeepLearning4J codebase. This tool is crucial for identifying regressions between different versions of DeepLearning4J by analyzing the execution patterns and performance metrics of ND4J operations.

## Features

- Records ND4J operation executions in real-time
- Stores execution data in an H2 database for efficient querying
- Indexes the specified DeepLearning4J codebase for reference
- Can be injected into a running DeepLearning4J application as a Java agent
- Provides methods for querying and analyzing recorded operations
- Includes a StackTraceCodeFinder utility for locating source code lines
- Features a JsonComparisonReport tool for comparing operation logs between different runs or versions
- Offers a JsonReport utility for exporting operation logs to JSON format

## Prerequisites

- Java 8 or higher
- Maven 3.x
- Access to the DeepLearning4J codebase you want to analyze

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-repo/nd4j-log-analyzer.git
   ```

2. Navigate to the project directory:
   ```
   cd nd4j-log-analyzer
   ```

3. Build the project using Maven:
   ```
   mvn clean package
   ```

## Usage

To use the ND4J Log Analyzer, you need to inject it as a Java agent into your DeepLearning4J application. Use the following VM arguments when running your Java application:

```
-DsourceCodeIndexerPath=/path/to/your/deeplearning4j/codebase -javaagent:/path/to/nd4j-log-analyzer-1.0-SNAPSHOT.jar
```

Example:
```
-DsourceCodeIndexerPath=/home/user/Documents/GitHub/deeplearning4j/ -javaagent:/home/user/Documents/GitHub/deeplearning4j/contrib/nd4j-log-analyzer/nd4j-log-analyzer/target/nd4j-log-analyzer-1.0-SNAPSHOT.jar
```

Make sure to replace the paths with the appropriate locations on your system.

## Configuration

The agent uses two main configuration options:

1. `sourceCodeIndexerPath`: The path to the DeepLearning4J codebase you want to index.
2. `javaagent`: The path to the compiled ND4J Log Analyzer JAR file.

## Database Structure

The H2 database is automatically created and managed by the agent. It contains two main tables:

1. `OpLogEvent`: Stores information about ND4J operation executions.
    - Columns: id, opName, inputs, outputs, stackTrace, sourceCodeLine

2. `SourceCodeLine`: Stores indexed source code information (created only if `sourceCodeIndexerPath` is provided).
    - Columns: id, packageName, className, lineNumber, line, fileName, lastUpdated

## Data Storage

- Operation logs are stored in the `OpLogEvent` table.
- Each operation execution is recorded with its name, inputs, outputs, stack trace, and corresponding source code line.
- Inputs and outputs are stored as arrays of strings.
- The source code indexer (if enabled) stores relevant code lines in the `SourceCodeLine` table.

## StackTraceCodeFinder Utility

The ND4J Log Analyzer includes a StackTraceCodeFinder utility that helps locate the relevant source code lines for recorded operations. Key features include:

- Resolves the source file path for a given fully qualified class name
- Retrieves the specific line of code from a stack trace element
- Caches file paths for improved performance
- Skips certain packages to focus on relevant code (configurable skip patterns)
- Searches for source roots within the specified directory

### Usage of StackTraceCodeFinder

```java
String rootDirectory = "/path/to/your/deeplearning4j/codebase";
StackTraceElement[] stackTrace = // ... obtain stack trace
String sourceCodeLine = StackTraceCodeFinder.getFirstLineOfCode(rootDirectory, stackTrace);
```

This utility is used internally by the Log Analyzer to associate recorded operations with their corresponding source code lines.

## JsonComparisonReport Utility

The JsonComparisonReport is a powerful tool for comparing operation logs between different runs or versions of your DeepLearning4J application. It helps identify differences in ND4J operations, which is crucial for detecting regressions or unexpected changes in behavior.

Key features of the JsonComparisonReport include:

- Compares operation logs stored in JSON format from two different directories
- Supports multiple epsilon values for floating-point comparisons
- Generates detailed reports of differences found between operation logs
- Identifies the earliest difference in the execution flow
- Filters differences based on a specified epsilon threshold

### Usage of JsonComparisonReport

To use the JsonComparisonReport, run it as a standalone Java application:

```
java org.nd4j.interceptor.data.JsonComparisonReport <directory1> <directory2>
```

Where:
- `<directory1>` is the path to the first set of JSON log files
- `<directory2>` is the path to the second set of JSON log files to compare against

The tool will generate two types of reports for each epsilon value defined in `InterceptorEnvironment.EPSILONS`:

1. `comparison_report_<epsilon>.json`: A detailed report of all differences found
2. `earliest_difference_<epsilon>.json`: Information about the first difference encountered in the execution flow

These reports can be used to identify and analyze discrepancies between different runs or versions of your DeepLearning4J application.

## JsonReport Utility

The JsonReport is a utility tool that generates JSON files for each unique operation name from the recorded ND4J operations. This tool is useful for exporting the collected data in a format that's easy to analyze or compare using other tools.

Key features of the JsonReport include:

- Generates a separate JSON file for each unique operation name
- Groups operation log events by source code line
- Uses a custom ObjectMapper for proper serialization of JSON arrays
- Creates a new directory for storing the generated JSON reports

### Usage of JsonReport

To use the JsonReport, run it as a standalone Java application:

```
java org.nd4j.interceptor.data.JsonReport <path_to_oplog.db>
```

Where:
- `<path_to_oplog.db>` is the path to the H2 database file containing the recorded operations

The tool will create a new directory called "jsonReports" (or clear it if it already exists) and generate JSON files for each unique operation name found in the database.

### Output

For each unique operation name, a JSON file will be created in the "jsonReports" directory. The file name will be `<operation_name>.json`. Each file contains:

- Grouped operation log events by source code line
- Detailed information about each operation execution, including inputs, outputs, and stack traces

These JSON files can be used for further analysis, comparison between different runs, or as input for other tools like the JsonComparisonReport.

## Workflow for Analyzing ND4J Operations

1. Run your DeepLearning4J application with the ND4J Log Analyzer agent to collect operation data.
2. Use the JsonReport utility to export the collected data to JSON files:
   ```
   java org.nd4j.interceptor.data.JsonReport path/to/your/oplog.db
   ```
3. If you want to compare two different runs or versions:
   a. Generate JSON reports for both runs using JsonReport
   b. Use the JsonComparisonReport to compare the generated JSON files:
      ```
      java org.nd4j.interceptor.data.JsonComparisonReport path/to/jsonReports1 path/to/jsonReports2
      ```
4. Analyze the comparison reports to identify differences or potential regressions in ND4J operations.

## Analyzing Results

After running your DeepLearning4J application with the agent, you can query the H2 database to analyze the recorded operations. The `InterceptorPersistence` class provides several methods for data analysis:

1. Get all unique operation names:
   ```java
   Set<String> uniqueOpNames = InterceptorPersistence.getUniqueOpNames(filePath);
   ```

2. Filter operations by name:
   ```java
   List<OpLogEvent> filteredEvents = InterceptorPersistence.filterByOpName(filePath, opName);
   ```

3. Group operations by source code line:
   ```java
   Map<String, List<OpLogEvent>> groupedEvents = InterceptorPersistence.groupedByCodeSortedByEventId(logEvents);
   ```

You can also use your preferred SQL client or the H2 Console to connect to the database and run custom queries.

For comparing results between different runs or versions, use the JsonComparisonReport utility as described above.

For exporting the recorded operations to JSON format for further analysis or comparison, use the JsonReport utility as described in the previous section.

## Troubleshooting

- If the agent fails to attach, ensure that you have the correct paths specified in the VM arguments.
- Check the console output for any error messages from the agent.
- Verify that you have write permissions in the directory where the H2 database is being created.
- If tables are not created properly, you can use the `InterceptorPersistence.listTables()` method to check the existing tables in the database.
- If source code lines are not being found, check that the `sourceCodeIndexerPath` is correct and that the StackTraceCodeFinder can access the necessary files.
- When using the JsonComparisonReport, ensure that the JSON log files are in the correct format and located in the specified directories.
- When using the JsonReport, ensure that the path to the oplog.db file is correct and that you have write permissions in the directory where the JSON files will be created.

## Contributing

Contributions to the ND4J Log Analyzer are welcome! Please submit pull requests or open issues on the GitHub repository.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.