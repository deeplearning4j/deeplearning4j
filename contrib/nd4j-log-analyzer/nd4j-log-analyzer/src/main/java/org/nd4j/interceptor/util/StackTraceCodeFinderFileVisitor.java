package org.nd4j.interceptor.util;

import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;

public class StackTraceCodeFinderFileVisitor implements FileVisitor<Path> {
   public List<Path> sourceRoots = new ArrayList<>();

    @Override
    public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException {
        if (dir.endsWith("src/main/java") || dir.endsWith("src/test/java")) {
            sourceRoots.add(dir);
            return FileVisitResult.SKIP_SUBTREE;
        }
        return FileVisitResult.CONTINUE;
    }

    @Override
    public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
        return FileVisitResult.CONTINUE;
    }

    @Override
    public FileVisitResult visitFileFailed(Path file, IOException exc) throws IOException {
        return FileVisitResult.CONTINUE;
    }

    @Override
    public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
        return FileVisitResult.CONTINUE;
    }


    public static void main(String... args) {
        StackTraceCodeFinderFileVisitor visitor = new StackTraceCodeFinderFileVisitor();
        try {
            Files.walkFileTree(Paths.get("/home/agibsonccc/Documents/GitHub/deeplearning4j/"), visitor);
        } catch (IOException e) {
            e.printStackTrace();
        }
        for (Path p : visitor.sourceRoots) {
            System.out.println(p);
        }
    }

}
