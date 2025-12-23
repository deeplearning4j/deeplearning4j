package org.deeplearning4j.tools;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Command(name = "cpp-dependency-analyzer", 
         mixinStandardHelpOptions = true, 
         version = "1.0.0",
         description = "Analyzes C++ include dependencies")
public class CppDependencyAnalyzer implements Callable<Integer> {

    @Parameters(index = "0", description = "Root directory to analyze")
    private Path rootDirectory;

    @Option(names = {"-v", "--verbose"}, description = "Verbose output")
    private boolean verbose;

    private static final Pattern INCLUDE_PATTERN = Pattern.compile("^\\s*#\\s*include\\s+[\"<]([^\\s\">]+)[\">]");
    private static final Set<String> CPP_EXTENSIONS = Set.of(".cpp", ".cxx", ".cc", ".c", ".hpp", ".h", ".hxx", ".hXX", ".chpp", ".cu", ".cuh");

    private final Map<String, Set<String>> fileDependencies = new HashMap<>();
    private final Map<String, String> fileToModule = new HashMap<>();
    private final Set<String> allFiles = new HashSet<>();

    public static void main(String[] args) {
        System.exit(new CommandLine(new CppDependencyAnalyzer()).execute(args));
    }

    @Override
    public Integer call() throws Exception {
        if (!Files.exists(rootDirectory) || !Files.isDirectory(rootDirectory)) {
            System.err.println("Invalid directory: " + rootDirectory);
            return 1;
        }

        System.out.println("Analyzing: " + rootDirectory.toAbsolutePath());
        
        // Find all C++ files
        findAllFiles();
        
        // Analyze each file's includes
        analyzeIncludes();
        
        // Generate report
        generateReport();
        
        return 0;
    }

    private void findAllFiles() throws IOException {
        Files.walkFileTree(rootDirectory, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                String fileName = file.getFileName().toString();
                if (hasCppExtension(fileName)) {
                    String relativePath = rootDirectory.relativize(file).toString().replace('\\', '/');
                    allFiles.add(relativePath);
                    fileToModule.put(relativePath, getModule(relativePath));
                    fileDependencies.put(relativePath, new HashSet<>());
                }
                return FileVisitResult.CONTINUE;
            }
        });
        
        System.out.println("Found " + allFiles.size() + " C++ files");
    }

    private void analyzeIncludes() throws IOException {
        for (String filePath : allFiles) {
            Path fullPath = rootDirectory.resolve(filePath);
            analyzeFile(filePath, fullPath);
        }
    }

    private void analyzeFile(String filePath, Path fullPath) throws IOException {
        List<String> lines = Files.readAllLines(fullPath);
        Set<String> dependencies = fileDependencies.get(filePath);
        
        for (String line : lines) {
            // Remove line comments
            int commentPos = line.indexOf("//");
            if (commentPos >= 0) {
                line = line.substring(0, commentPos);
            }
            
            Matcher matcher = INCLUDE_PATTERN.matcher(line.trim());
            if (matcher.matches()) {
                String includePath = matcher.group(1);
                String resolved = resolveInclude(filePath, includePath, line.contains("\""));
                
                if (resolved != null) {
                    dependencies.add(resolved);
                    if (verbose) {
                        System.out.println(filePath + " -> " + resolved);
                    }
                }
            }
        }
    }

    private String resolveInclude(String sourceFile, String includePath, boolean isQuoted) {
        // If quoted include, try relative to source file first
        if (isQuoted) {
            Path sourcePath = Paths.get(sourceFile).getParent();
            if (sourcePath != null) {
                String candidate = sourcePath.resolve(includePath).normalize().toString().replace('\\', '/');
                if (allFiles.contains(candidate)) {
                    return candidate;
                }
            }
        }
        
        // Try relative to root
        String candidate = Paths.get(includePath).normalize().toString().replace('\\', '/');
        if (allFiles.contains(candidate)) {
            return candidate;
        }
        
        // Try common include paths
        String[] searchPaths = {"include/", "src/", "lib/", "tests/", "blas/"};
        for (String searchPath : searchPaths) {
            candidate = Paths.get(searchPath, includePath).normalize().toString().replace('\\', '/');
            if (allFiles.contains(candidate)) {
                return candidate;
            }
        }
        
        return null; // External or not found
    }

    private void generateReport() {
        System.out.println("\n=== DEPENDENCY ANALYSIS REPORT ===\n");
        
        // File-level dependencies
        System.out.println("FILE DEPENDENCIES:");
        System.out.println("-".repeat(50));
        
        Map<String, Integer> dependencyCount = new HashMap<>();
        for (Map.Entry<String, Set<String>> entry : fileDependencies.entrySet()) {
            String file = entry.getKey();
            Set<String> deps = entry.getValue();
            
            dependencyCount.put(file, deps.size());
            
            if (!deps.isEmpty()) {
                System.out.println(file + " depends on:");
                for (String dep : deps.stream().sorted().toArray(String[]::new)) {
                    System.out.println("  -> " + dep);
                }
                System.out.println();
            }
        }
        
        // Module-level dependencies
        System.out.println("\nMODULE DEPENDENCIES:");
        System.out.println("-".repeat(50));
        
        Map<String, Set<String>> moduleDeps = new HashMap<>();
        for (Map.Entry<String, Set<String>> entry : fileDependencies.entrySet()) {
            String sourceModule = fileToModule.get(entry.getKey());
            Set<String> targetModules = moduleDeps.computeIfAbsent(sourceModule, k -> new HashSet<>());
            
            for (String depFile : entry.getValue()) {
                String targetModule = fileToModule.get(depFile);
                if (targetModule != null && !targetModule.equals(sourceModule)) {
                    targetModules.add(targetModule);
                }
            }
        }
        
        for (String module : moduleDeps.keySet().stream().sorted().toArray(String[]::new)) {
            Set<String> deps = moduleDeps.get(module);
            if (!deps.isEmpty()) {
                System.out.println(module + " depends on:");
                for (String dep : deps.stream().sorted().toArray(String[]::new)) {
                    System.out.println("  -> " + dep);
                }
                System.out.println();
            }
        }
        
        // Summary statistics
        System.out.println("\nSUMMARY:");
        System.out.println("-".repeat(50));
        
        Map<String, Long> moduleFileCounts = new HashMap<>();
        for (String file : allFiles) {
            String module = fileToModule.get(file);
            moduleFileCounts.merge(module, 1L, Long::sum);
        }
        
        System.out.println("Modules and file counts:");
        for (Map.Entry<String, Long> entry : moduleFileCounts.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
                .toArray(Map.Entry[]::new)) {
            System.out.println("  " + entry.getKey() + ": " + entry.getValue() + " files");
        }
        
        int totalDeps = dependencyCount.values().stream().mapToInt(Integer::intValue).sum();
        System.out.println("\nTotal internal dependencies: " + totalDeps);
        System.out.println("Average dependencies per file: " + (totalDeps / (double) allFiles.size()));
    }

    private boolean hasCppExtension(String fileName) {
        int lastDot = fileName.lastIndexOf('.');
        if (lastDot > 0) {
            String ext = fileName.substring(lastDot);
            return CPP_EXTENSIONS.contains(ext);
        }
        return false;
    }

    private String getModule(String filePath) {
        int firstSlash = filePath.indexOf('/');
        if (firstSlash > 0) {
            return filePath.substring(0, firstSlash);
        }
        return "root";
    }
}
