import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Stream;

/**
 * Locates exact UTF-8 constants in every class on a resolved runtime classpath.
 *
 * <p>This is intentionally a standalone source-file program. The Android AOT
 * producer can preserve its exact classpath and this audit can inspect those
 * bytes without adding diagnostic classes to the native image.</p>
 */
public final class EmbeddedClasspathAudit {
    private EmbeddedClasspathAudit() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: java EmbeddedClasspathAudit.java CLASSPATH_FILE UTF8_CONSTANT...");
            System.exit(2);
        }

        Path classpathFile = Path.of(args[0]).toAbsolutePath().normalize();
        List<byte[]> needles = new ArrayList<>();
        for (int i = 1; i < args.length; i++) {
            needles.add(args[i].getBytes(StandardCharsets.UTF_8));
        }

        String classpath = Files.readString(classpathFile, StandardCharsets.UTF_8).trim();
        if (classpath.isEmpty()) {
            throw new IllegalArgumentException("Classpath file is empty: " + classpathFile);
        }

        List<String> matches = new ArrayList<>();
        for (String entry : classpath.split(java.util.regex.Pattern.quote(File.pathSeparator))) {
            if (entry.isBlank()) {
                continue;
            }
            Path path = Path.of(entry).toAbsolutePath().normalize();
            if (Files.isDirectory(path)) {
                scanDirectory(path, needles, matches);
            } else if (Files.isRegularFile(path) && path.getFileName().toString().endsWith(".jar")) {
                scanJar(path, needles, matches);
            }
        }

        matches.stream().sorted().forEach(System.out::println);
    }

    private static void scanDirectory(Path root, List<byte[]> needles, List<String> matches)
            throws IOException {
        try (Stream<Path> paths = Files.walk(root)) {
            for (Path file : paths.filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().endsWith(".class"))
                    .sorted()
                    .toList()) {
                recordMatches(
                        root.toString(),
                        root.relativize(file).toString().replace(File.separatorChar, '/'),
                        Files.readAllBytes(file),
                        needles,
                        matches);
            }
        }
    }

    private static void scanJar(Path jarPath, List<byte[]> needles, List<String> matches)
            throws IOException {
        try (JarFile jar = new JarFile(jarPath.toFile(), false)) {
            List<JarEntry> entries = jar.stream()
                    .filter(entry -> !entry.isDirectory() && entry.getName().endsWith(".class"))
                    .sorted(Comparator.comparing(JarEntry::getName))
                    .toList();
            for (JarEntry entry : entries) {
                try (InputStream input = jar.getInputStream(entry)) {
                    recordMatches(
                            jarPath.toString(),
                            entry.getName(),
                            input.readAllBytes(),
                            needles,
                            matches);
                }
            }
        }
    }

    private static void recordMatches(
            String container,
            String className,
            byte[] bytes,
            List<byte[]> needles,
            List<String> matches) {
        List<String> matched = new ArrayList<>();
        for (byte[] needle : needles) {
            if (contains(bytes, needle)) {
                matched.add(new String(needle, StandardCharsets.UTF_8));
            }
        }
        if (!matched.isEmpty()) {
            matches.add(container + "\t" + className + "\t" + String.join(",", matched));
        }
    }

    private static boolean contains(byte[] haystack, byte[] needle) {
        if (needle.length == 0) {
            return true;
        }
        for (int start = 0; start <= haystack.length - needle.length; start++) {
            if (haystack[start] == needle[0]
                    && Arrays.equals(
                            haystack,
                            start,
                            start + needle.length,
                            needle,
                            0,
                            needle.length)) {
                return true;
            }
        }
        return false;
    }
}
