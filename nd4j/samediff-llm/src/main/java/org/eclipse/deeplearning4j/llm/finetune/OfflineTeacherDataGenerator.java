package org.eclipse.deeplearning4j.llm.finetune;

import org.eclipse.deeplearning4j.llm.generation.GenerationPipeline;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Runs a teacher offline and returns validated sequence-distillation examples. */
public class OfflineTeacherDataGenerator {
    @FunctionalInterface
    public interface TeacherGenerator {
        String generate(String prompt, int maxNewTokens);
    }

    private final TeacherGenerator teacher;
    private final CanonicalContextSerializer serializer;
    private final List<TeacherOutputValidator> validators;
    private final int maxNewTokens;

    public OfflineTeacherDataGenerator(TeacherGenerator teacher, CanonicalContextSerializer serializer,
                                       List<TeacherOutputValidator> validators, int maxNewTokens) {
        if (teacher == null) throw new IllegalArgumentException("teacher is required");
        if (maxNewTokens < 1) throw new IllegalArgumentException("maxNewTokens must be positive");
        this.teacher = teacher;
        this.serializer = serializer == null ? new CanonicalContextSerializer() : serializer;
        this.validators = validators == null ? new ArrayList<>() : new ArrayList<>(validators);
        this.maxNewTokens = maxNewTokens;
    }

    public static TeacherGenerator fromPipeline(GenerationPipeline pipeline) {
        return (prompt, maxTokens) -> pipeline.generate(prompt, maxTokens).getText();
    }

    public GeneratedTrainingExample generate(TeacherExampleRequest request) {
        request.validate();
        String canonicalPrompt = serializer.serialize(request.getContext());
        String teacherPrompt = buildTeacherPrompt(request.getSystemPrompt(), canonicalPrompt);
        String response = clean(teacher.generate(teacherPrompt, maxNewTokens));
        if (response.isEmpty()) throw new IllegalArgumentException("Teacher returned an empty response for " + request.getId());

        List<String> rejectionReasons = new ArrayList<>();
        for (TeacherOutputValidator validator : validators) {
            TeacherValidationResult result = validator.validate(request, response);
            if (result != null && !result.isAccepted()) rejectionReasons.addAll(result.getReasons());
        }
        if (!rejectionReasons.isEmpty()) {
            throw new IllegalArgumentException("Teacher response rejected for " + request.getId() + ": " + rejectionReasons);
        }

        GeneratedTrainingExample example = new GeneratedTrainingExample();
        example.setId(request.getId());
        example.setSystemPrompt(request.getSystemPrompt());
        example.setPrompt(canonicalPrompt);
        example.setResponse(response);
        example.setContext(new LinkedHashMap<>(request.getContext()));
        example.setMetadata(request.getMetadata() == null ? new LinkedHashMap<>() : new LinkedHashMap<>(request.getMetadata()));
        return example;
    }

    public List<GeneratedTrainingExample> generateAll(List<TeacherExampleRequest> requests) {
        List<GeneratedTrainingExample> result = new ArrayList<>(requests.size());
        for (TeacherExampleRequest request : requests) result.add(generate(request));
        return result;
    }

    static String buildTeacherPrompt(String systemPrompt, String canonicalPrompt) {
        if (systemPrompt == null || systemPrompt.trim().isEmpty()) return canonicalPrompt;
        return systemPrompt.trim() + "\n\n" + canonicalPrompt;
    }

    static String clean(String text) {
        if (text == null) return "";
        String result = text.trim().replaceFirst("(?i)^(assistant|response|output)\\s*:\\s*", "");
        return result;
    }
}
