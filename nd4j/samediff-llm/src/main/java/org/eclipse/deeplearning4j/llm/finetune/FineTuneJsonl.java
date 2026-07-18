package org.eclipse.deeplearning4j.llm.finetune;

import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

/** Strict JSONL persistence for teacher requests and generated training examples. */
public final class FineTuneJsonl {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private FineTuneJsonl() {}

    public static <T> List<T> read(File file, Class<T> type) throws IOException {
        try (InputStream stream = Files.newInputStream(file.toPath())) {
            return read(stream, type);
        }
    }

    public static <T> List<T> read(InputStream stream, Class<T> type) throws IOException {
        List<T> result = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            String line;
            int lineNumber = 0;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                if (line.trim().isEmpty()) continue;
                try {
                    result.add(MAPPER.readValue(line, type));
                } catch (Exception e) {
                    throw new IOException("Malformed JSONL at line " + lineNumber, e);
                }
            }
        }
        return result;
    }

    public static void write(File file, List<?> records) throws IOException {
        if (file.toPath().getParent() != null) Files.createDirectories(file.toPath().getParent());
        try (OutputStream stream = Files.newOutputStream(file.toPath())) {
            write(stream, records);
        }
    }

    public static void write(OutputStream stream, List<?> records) throws IOException {
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(stream, StandardCharsets.UTF_8));
        for (Object record : records) {
            writer.write(MAPPER.writeValueAsString(record));
            writer.newLine();
        }
        writer.flush();
    }

    /** Appends and flushes one record so long-running generation jobs are restartable. */
    public static void append(File file, Object record) throws IOException {
        if (file.toPath().getParent() != null) Files.createDirectories(file.toPath().getParent());
        try (BufferedWriter writer = Files.newBufferedWriter(file.toPath(), StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            writer.write(MAPPER.writeValueAsString(record));
            writer.newLine();
        }
    }
}
