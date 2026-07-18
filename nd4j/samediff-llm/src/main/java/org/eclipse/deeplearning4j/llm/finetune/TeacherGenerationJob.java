package org.eclipse.deeplearning4j.llm.finetune;

import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;

/**
 * Runs teacher generation as an append-only, resumable job.
 *
 * <p>Accepted and fallback IDs are skipped on resume. Rejections are audit records only and
 * are retried on the next run.</p>
 */
public final class TeacherGenerationJob {
    private final OfflineTeacherDataGenerator generator;

    public TeacherGenerationJob(OfflineTeacherDataGenerator generator) {
        if (generator == null) throw new IllegalArgumentException("generator is required");
        this.generator = generator;
    }

    public TeacherGenerationReport run(List<TeacherExampleRequest> requests, File output) throws IOException {
        return run(requests, output, null, null, null);
    }

    public TeacherGenerationReport run(List<TeacherExampleRequest> requests, File output,
                                       File rejectionOutput, TeacherFallbackProvider fallbackProvider,
                                       String runId) throws IOException {
        if (requests == null) throw new IllegalArgumentException("requests are required");
        if (output == null) throw new IllegalArgumentException("output is required");

        Set<String> completed = new HashSet<>();
        if (output.isFile()) {
            for (GeneratedTrainingExample example :
                    FineTuneJsonl.read(output, GeneratedTrainingExample.class)) {
                example.validate();
                if (!completed.add(example.getId())) {
                    throw new IOException("Duplicate generated example id: " + example.getId());
                }
            }
        }

        TeacherGenerationReport report = new TeacherGenerationReport();
        for (TeacherExampleRequest request : requests) {
            String id = request == null ? null : request.getId();
            if (id != null && completed.contains(id)) {
                report.skip();
                continue;
            }
            try {
                GeneratedTrainingExample example = generator.generate(request);
                persist(output, completed, example);
                report.accept(example);
            } catch (RuntimeException teacherFailure) {
                String reason = message(teacherFailure);
                if (fallbackProvider != null) {
                    try {
                        GeneratedTrainingExample fallback = fallbackProvider.fallback(request, reason);
                        fallback.validate();
                        if (!fallback.getId().equals(id)) {
                            throw new IllegalArgumentException("Fallback id must match request id");
                        }
                        if (fallback.getMetadata() == null) fallback.setMetadata(new LinkedHashMap<>());
                        fallback.getMetadata().put("generationSource", "fallback");
                        fallback.getMetadata().put("teacherRejectionReason", reason);
                        persist(output, completed, fallback);
                        report.acceptFallback(fallback);
                        continue;
                    } catch (RuntimeException fallbackFailure) {
                        reason += "; fallback failed: " + message(fallbackFailure);
                    }
                }
                report.reject(id, reason);
                if (rejectionOutput != null) {
                    FineTuneJsonl.append(rejectionOutput,
                            new TeacherGenerationRejection(id, reason, 1, runId, request));
                }
            }
        }
        return report;
    }

    private static void persist(File output, Set<String> completed,
                                GeneratedTrainingExample example) throws IOException {
        if (!completed.add(example.getId())) {
            throw new IllegalArgumentException("Duplicate generated example id: " + example.getId());
        }
        FineTuneJsonl.append(output, example);
    }

    private static String message(RuntimeException e) {
        return e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
    }
}
