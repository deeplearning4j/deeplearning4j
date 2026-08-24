/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.vlm.output.protocol;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.llm.config.ModelConfig;
import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.tokenizer.Encoding;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.eclipse.deeplearning4j.vlm.output.DocTagsParser;
import org.eclipse.deeplearning4j.vlm.output.DocumentStructure;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.ServiceLoader;

/** Resolves package-declared output protocols and external providers without model-name checks. */
@Slf4j
public final class VlmOutputProtocolRegistry {
    public static final String MANIFEST_NAME = "vlm-output-protocol.json";

    private final VlmProtocolDefinition definition;
    private final Map<String, VlmOutputProtocolProvider> providers;

    private VlmOutputProtocolRegistry(VlmProtocolDefinition definition,
                                      Map<String, VlmOutputProtocolProvider> providers) {
        this.definition = definition;
        this.providers = providers;
    }

    public static VlmOutputProtocolRegistry load(File modelDirectory) throws IOException {
        VlmProtocolDefinition definition = new VlmProtocolDefinition();
        File manifest = modelDirectory == null ? null : new File(modelDirectory, MANIFEST_NAME);
        if (manifest != null && manifest.isFile()) {
            definition = new ObjectMapper().readValue(manifest, VlmProtocolDefinition.class);
            if (definition.getSchemaVersion() != 1) {
                throw new IOException("Unsupported VLM output protocol schemaVersion="
                        + definition.getSchemaVersion() + " in " + manifest);
            }
            validateDefinition(definition, manifest, discoverProviders());
        }
        return new VlmOutputProtocolRegistry(definition, discoverProviders());
    }

    public static VlmOutputProtocolRegistry fallback() {
        return new VlmOutputProtocolRegistry(new VlmProtocolDefinition(), discoverProviders());
    }

    public VlmOutputProtocol resolve(String requestedProtocolId) {
        String selected = nonBlank(requestedProtocolId) ? requestedProtocolId : definition.getDefaultProtocol();
        VlmProtocolDefinition.Protocol protocolDefinition = selected == null
                ? null : definition.getProtocols().get(selected);
        if (protocolDefinition == null && selected != null && selected.startsWith("builtin.")) {
            protocolDefinition = new VlmProtocolDefinition.Protocol();
            protocolDefinition.setProvider(selected);
        }
        if (protocolDefinition == null) {
            if (nonBlank(requestedProtocolId)) {
                throw new IllegalArgumentException("Unknown VLM output protocol: " + requestedProtocolId);
            }
            selected = "fallback";
            protocolDefinition = new VlmProtocolDefinition.Protocol();
            protocolDefinition.setProvider("builtin.plain");
        }
        VlmOutputProtocolProvider provider = providers.get(protocolDefinition.getProvider());
        if (provider == null) {
            throw new IllegalArgumentException("Unknown VLM output protocol provider: "
                    + protocolDefinition.getProvider());
        }
        return provider.bind(selected, protocolDefinition);
    }

    public VlmProtocolPlan prepare(VlmProtocolRequest request, Tokenizer tokenizer, ModelConfig config) {
        VlmProtocolRequest effective = request == null ? VlmProtocolRequest.builder().build() : request;
        return resolve(effective.getProtocolId()).prepare(effective, tokenizer, config);
    }

    public VlmProtocolOutput process(VlmProtocolRequest request, VlmProtocolPlan plan,
                                     GenerationResult generation, Tokenizer tokenizer) {
        VlmProtocolRequest effective = request == null ? VlmProtocolRequest.builder().build() : request;
        return resolve(effective.getProtocolId()).process(effective, plan, generation, tokenizer);
    }

    public GenerationResult mergeRegions(VlmProtocolRequest request, VlmProtocolPlan plan,
                                         List<GenerationResult> regions, Tokenizer tokenizer) {
        VlmProtocolRequest effective = request == null ? VlmProtocolRequest.builder().build() : request;
        return resolve(effective.getProtocolId()).mergeRegions(effective, plan, regions, tokenizer);
    }

    public String defaultProtocolId() {
        return definition.getDefaultProtocol();
    }

    private static Map<String, VlmOutputProtocolProvider> discoverProviders() {
        Map<String, VlmOutputProtocolProvider> result = new LinkedHashMap<>();
        register(result, new GenericProvider("builtin.plain", Mode.PLAIN));
        register(result, new GenericProvider("builtin.markup", Mode.MARKUP));
        register(result, new GenericProvider("builtin.json", Mode.JSON));
        register(result, new GenericProvider("builtin.tagged", Mode.TAGGED));
        register(result, new DocTagsProvider());
        for (VlmOutputProtocolProvider provider : ServiceLoader.load(VlmOutputProtocolProvider.class)) {
            register(result, provider);
        }
        return Collections.unmodifiableMap(result);
    }

    private static void validateDefinition(VlmProtocolDefinition definition, File manifest,
                                           Map<String, VlmOutputProtocolProvider> providers) throws IOException {
        if (definition.getSchemaVersion() != 1) {
            throw new IOException("Unsupported VLM protocol manifest schemaVersion "
                    + definition.getSchemaVersion() + ": " + manifest);
        }
        if (definition.getProtocols() == null || definition.getProtocols().isEmpty()) {
            throw new IOException("VLM protocol manifest has no protocols: " + manifest);
        }
        if (!nonBlank(definition.getDefaultProtocol())
                || !definition.getProtocols().containsKey(definition.getDefaultProtocol())) {
            throw new IOException("VLM protocol manifest defaultProtocol is missing or unknown: " + manifest);
        }
        for (Map.Entry<String, VlmProtocolDefinition.Protocol> entry : definition.getProtocols().entrySet()) {
            VlmProtocolDefinition.Protocol protocol = entry.getValue();
            if (!nonBlank(entry.getKey()) || protocol == null || !nonBlank(protocol.getProvider())
                    || !providers.containsKey(protocol.getProvider())) {
                throw new IOException("Protocol '" + entry.getKey() + "' has an unknown provider in " + manifest);
            }
            if (protocol.getTasks() == null || protocol.getTermination() == null
                    || protocol.getCompletion() == null || protocol.getOutput() == null) {
                throw new IOException("Protocol '" + entry.getKey() + "' has null sections in " + manifest);
            }
            if (!protocol.getTasks().isEmpty()
                    && (!nonBlank(protocol.getDefaultTask())
                    || !protocol.getTasks().containsKey(protocol.getDefaultTask()))) {
                throw new IOException("Protocol '" + entry.getKey() + "' has an unknown defaultTask in " + manifest);
            }
            for (Map.Entry<String, VlmProtocolDefinition.Task> taskEntry : protocol.getTasks().entrySet()) {
                VlmProtocolDefinition.Task task = taskEntry.getValue();
                if (!nonBlank(taskEntry.getKey()) || task == null
                        || !("RAW".equalsIgnoreCase(task.getFraming())
                        || "CHAT_TEMPLATE".equalsIgnoreCase(task.getFraming()))) {
                    throw new IOException("Protocol '" + entry.getKey() + "' has an invalid task '"
                            + taskEntry.getKey() + "' in " + manifest);
                }
                if (task.getAliases() != null) {
                    for (String alias : task.getAliases()) {
                        if (!nonBlank(alias)) {
                            throw new IOException("Protocol '" + entry.getKey()
                                    + "' has a blank task alias in " + manifest);
                        }
                    }
                }
            }
            if (protocol.getTermination().getSequences() != null) {
                for (VlmProtocolDefinition.Stop stop : protocol.getTermination().getSequences()) {
                    if (stop == null || !nonBlank(stop.getId()) || (!nonBlank(stop.getText())
                            && (stop.getTokenIds() == null || stop.getTokenIds().length == 0))) {
                        throw new IOException("Protocol '" + entry.getKey() + "' has an invalid stop in " + manifest);
                    }
                    try {
                        VlmStopSequence.Kind.valueOf(stop.getKind().toUpperCase(Locale.ROOT));
                        VlmStopSequence.Retention.valueOf(stop.getRetention().toUpperCase(Locale.ROOT));
                    } catch (Exception invalid) {
                        throw new IOException("Protocol '" + entry.getKey() + "' has invalid stop metadata in " + manifest, invalid);
                    }
                    if (stop.getTokenIds() != null) {
                        for (int tokenId : stop.getTokenIds()) {
                            if (tokenId < 0) {
                                throw new IOException("Protocol '" + entry.getKey()
                                        + "' has a negative stop token ID in " + manifest);
                            }
                        }
                    }
                }
            }
            if (!nonBlank(protocol.getOutput().getNativeFormat())
                    || protocol.getOutput().getRenderers() == null) {
                throw new IOException("Protocol '" + entry.getKey() + "' has no native output format in " + manifest);
            }
            for (Map.Entry<String, String> renderer : protocol.getOutput().getRenderers().entrySet()) {
                try {
                    VlmRenderFormat.valueOf(renderer.getKey().toUpperCase(Locale.ROOT));
                } catch (Exception invalid) {
                    throw new IOException("Protocol '" + entry.getKey()
                            + "' has an unknown render format '" + renderer.getKey() + "' in " + manifest, invalid);
                }
                if (!nonBlank(renderer.getValue())) {
                    throw new IOException("Protocol '" + entry.getKey()
                            + "' has a blank renderer for '" + renderer.getKey() + "' in " + manifest);
                }
            }
        }
    }

    private static void register(Map<String, VlmOutputProtocolProvider> providers,
                                 VlmOutputProtocolProvider provider) {
        VlmOutputProtocolProvider prior = providers.putIfAbsent(provider.providerId(), provider);
        if (prior != null && prior.getClass() != provider.getClass()) {
            throw new IllegalStateException("Duplicate VLM output protocol provider: " + provider.providerId());
        }
    }

    private static boolean nonBlank(String value) {
        return value != null && !value.isBlank();
    }

    private enum Mode { PLAIN, MARKUP, JSON, TAGGED }

    private static class GenericProvider implements VlmOutputProtocolProvider {
        private final String id;
        private final Mode mode;

        private GenericProvider(String id, Mode mode) {
            this.id = id;
            this.mode = mode;
        }

        @Override public String providerId() { return id; }

        @Override
        public VlmOutputProtocol bind(String protocolId, VlmProtocolDefinition.Protocol definition) {
            return new BoundProtocol(protocolId, definition, mode);
        }
    }

    private static class BoundProtocol implements VlmOutputProtocol {
        protected final String protocolId;
        protected final VlmProtocolDefinition.Protocol definition;
        protected final Mode mode;

        private BoundProtocol(String protocolId, VlmProtocolDefinition.Protocol definition, Mode mode) {
            this.protocolId = protocolId;
            this.definition = definition;
            this.mode = mode;
        }

        @Override public String id() { return protocolId; }

        @Override
        public VlmProtocolPlan prepare(VlmProtocolRequest request, Tokenizer tokenizer,
                                       ModelConfig modelConfig) {
            VlmProtocolDefinition.Task task = resolveTask(request.getTask());
            String prompt = request.getPromptOverride() != null ? request.getPromptOverride()
                    : task != null && task.getPrompt() != null ? task.getPrompt()
                    : "Describe the image or document.";
            boolean chat = task == null || !"RAW".equalsIgnoreCase(task.getFraming());
            return VlmProtocolPlan.builder()
                    .protocolId(protocolId)
                    .task(taskName(request.getTask()))
                    .prompt(prompt)
                    .applyChatTemplate(chat)
                    .stops(resolveStopSequences(tokenizer))
                    .inheritModelEos(definition.getTermination() == null
                            || definition.getTermination().isInheritModelEos())
                    .inheritChatTemplateStops(definition.getTermination() == null
                            || definition.getTermination().isInheritChatTemplateStops())
                    .nativeFormat(definition.getOutput().getNativeFormat())
                    .structuralCompletionRequired(definition.getCompletion().isRequired())
                    .build();
        }

        @Override
        public VlmProtocolOutput process(VlmProtocolRequest request, VlmProtocolPlan plan,
                                         GenerationResult generation, Tokenizer tokenizer) {
            String raw = protocolText(plan, generation, tokenizer);
            VlmCompletion completion = assess(raw, generation);
            VlmStopSequence matched = matchedStop(plan, generation);
            return VlmProtocolOutput.builder()
                    .protocolId(protocolId)
                    .nativeFormat(definition.getOutput().getNativeFormat())
                    .rawText(raw)
                    .renderedText(render(request, raw))
                    .completion(completion)
                    .metadata(stopMetadata(matched))
                    .build();
        }

        @Override
        public GenerationResult mergeRegions(VlmProtocolRequest request, VlmProtocolPlan plan,
                                             List<GenerationResult> regions, Tokenizer tokenizer) {
            List<GenerationResult> processed = new ArrayList<>();
            if (regions != null) for (GenerationResult region : regions) {
                if (region == null) continue;
                processed.add(protocolGeneration(plan, region, tokenizer));
            }
            if (mode == Mode.JSON) {
                try {
                    org.nd4j.shade.jackson.databind.node.ArrayNode array = new ObjectMapper().createArrayNode();
                    for (GenerationResult region : processed) array.add(new ObjectMapper().readTree(region.getText()));
                    GenerationResult aggregate = mergeRegionResults(processed, null, null);
                    return aggregate.toBuilder().text(array.toString()).build();
                } catch (Exception invalid) {
                    throw new IllegalStateException("Unable to merge JSON VLM regions", invalid);
                }
            }
            return mergeRegionResults(processed, null, null);
        }

        private String render(VlmProtocolRequest request, String raw) {
            VlmRenderFormat requested = request == null || request.getRenderFormat() == null
                    ? VlmRenderFormat.PLAIN_TEXT : request.getRenderFormat();
            if (requested == VlmRenderFormat.RAW) return raw;
            String nativeFormat = definition.getOutput().getNativeFormat().toUpperCase(Locale.ROOT);
            boolean identity = mode == Mode.PLAIN
                    && (requested == VlmRenderFormat.PLAIN_TEXT || requested == VlmRenderFormat.MARKDOWN)
                    || mode == Mode.MARKUP && requested == VlmRenderFormat.MARKDOWN
                    && (nativeFormat.contains("MARKDOWN") || nativeFormat.contains("MARKUP")
                    || nativeFormat.contains("MMD") || nativeFormat.contains("LATEX"))
                    || mode == Mode.JSON && requested == VlmRenderFormat.JSON;
            String renderer = definition.getOutput().getRenderers().get(requested.name());
            if (identity || "RAW".equalsIgnoreCase(renderer) || "IDENTITY".equalsIgnoreCase(renderer)) {
                return raw;
            }
            throw new IllegalArgumentException("VLM protocol '" + protocolId + "' provider '"
                    + idForMode() + "' cannot render native format '" + nativeFormat
                    + "' as " + requested + "; request RAW or install a provider with that renderer");
        }

        private String idForMode() {
            switch (mode) {
                case MARKUP: return "builtin.markup";
                case JSON: return "builtin.json";
                case TAGGED: return "builtin.tagged";
                case PLAIN:
                default: return "builtin.plain";
            }
        }

        protected VlmCompletion assess(String raw, GenerationResult generation) {
            if (generation != null && generation.getFinishReason() == GenerationResult.FinishReason.MAX_TOKENS) {
                return VlmCompletion.builder().complete(false).usable(false)
                        .diagnostic("generation exhausted its context before a protocol terminator").build();
            }
            boolean complete = raw != null && !raw.isBlank();
            String diagnostic = complete ? "complete" : "empty model output";
            if (complete && mode == Mode.JSON) {
                try {
                    new ObjectMapper().readTree(raw);
                } catch (Exception invalidJson) {
                    complete = false;
                    diagnostic = "invalid JSON: " + invalidJson.getMessage();
                }
            } else if (complete && mode == Mode.TAGGED) {
                complete = tagsBalanced(raw);
                diagnostic = complete ? "complete" : "unbalanced tagged output";
            }
            boolean usable = complete || !definition.getCompletion().isRequired();
            return VlmCompletion.builder().complete(complete).usable(usable).diagnostic(diagnostic).build();
        }

        private VlmProtocolDefinition.Task resolveTask(String requestedTask) {
            String selected = nonBlank(requestedTask) ? requestedTask : definition.getDefaultTask();
            VlmProtocolDefinition.Task direct = definition.getTasks().get(selected);
            if (direct != null) return direct;
            for (Map.Entry<String, VlmProtocolDefinition.Task> entry : definition.getTasks().entrySet()) {
                List<String> aliases = entry.getValue().getAliases();
                if (aliases != null && aliases.contains(selected)) return entry.getValue();
            }
            if (nonBlank(requestedTask)) {
                throw new IllegalArgumentException("Unknown task '" + requestedTask
                        + "' for VLM output protocol '" + protocolId + "'");
            }
            return definition.getTasks().isEmpty() ? null : definition.getTasks().values().iterator().next();
        }

        private String taskName(String requestedTask) {
            return nonBlank(requestedTask) ? requestedTask : definition.getDefaultTask();
        }

        private List<VlmStopSequence> resolveStopSequences(Tokenizer tokenizer) {
            if (definition.getTermination() == null || definition.getTermination().getSequences() == null) {
                return Collections.emptyList();
            }
            List<VlmStopSequence> result = new ArrayList<>();
            for (VlmProtocolDefinition.Stop stop : definition.getTermination().getSequences()) {
                int[] ids = stop.getTokenIds();
                if ((ids == null || ids.length == 0) && nonBlank(stop.getText())) {
                    Integer atomic = tokenizer.getTokenId(stop.getText());
                    if (atomic != null) ids = new int[]{atomic};
                    else {
                        Encoding encoding = tokenizer.encode(stop.getText(), false);
                        ids = encoding == null ? null : encoding.getIds();
                    }
                }
                if (ids == null || ids.length == 0) {
                    throw new IllegalArgumentException("Protocol stop '" + stop.getId()
                            + "' could not be tokenized");
                }
                VlmStopSequence.Kind kind;
                VlmStopSequence.Retention retention;
                try { kind = VlmStopSequence.Kind.valueOf(stop.getKind().toUpperCase(Locale.ROOT)); }
                catch (Exception invalid) { throw new IllegalArgumentException("Invalid stop kind: " + stop.getKind()); }
                try { retention = VlmStopSequence.Retention.valueOf(stop.getRetention().toUpperCase(Locale.ROOT)); }
                catch (Exception invalid) { throw new IllegalArgumentException("Invalid stop retention: " + stop.getRetention()); }
                result.add(VlmStopSequence.builder().id(stop.getId()).kind(kind)
                        .retention(retention).tokenIds(ids.clone()).build());
            }
            return Collections.unmodifiableList(result);
        }

        protected static String protocolText(VlmProtocolPlan plan, GenerationResult generation,
                                             Tokenizer tokenizer) {
            if (generation == null || generation.getText() == null) return "";
            if (plan == null || generation.getFinishReason() != GenerationResult.FinishReason.STOP_SEQUENCE
                    || generation.getTokenIds() == null) return generation.getText();
            int[] generated = generation.getTokenIds();
            for (VlmStopSequence stop : plan.getStops()) {
                int[] sequence = stop.getTokenIds();
                if (stop.getRetention() != VlmStopSequence.Retention.DROP_MATCH
                        || sequence == null || sequence.length > generated.length) continue;
                int start = generated.length - sequence.length;
                boolean matches = true;
                for (int i = 0; i < sequence.length; i++) {
                    if (generated[start + i] != sequence[i]) { matches = false; break; }
                }
                if (matches) return tokenizer.decode(java.util.Arrays.copyOf(generated, start), false);
            }
            return generation.getText();
        }

        protected static GenerationResult protocolGeneration(VlmProtocolPlan plan,
                                                              GenerationResult generation,
                                                              Tokenizer tokenizer) {
            String text = protocolText(plan, generation, tokenizer);
            VlmStopSequence matched = matchedStop(plan, generation);
            if (matched == null || matched.getRetention() != VlmStopSequence.Retention.DROP_MATCH
                    || generation.getTokenIds() == null) {
                return generation.toBuilder().text(text).build();
            }
            int retainedLength = generation.getTokenIds().length - matched.getTokenIds().length;
            int[] retained = java.util.Arrays.copyOf(generation.getTokenIds(), Math.max(0, retainedLength));
            int generatedCount = Math.max(0,
                    generation.getGeneratedTokenCount() - matched.getTokenIds().length);
            return generation.toBuilder().text(text).tokenIds(retained)
                    .generatedTokenCount(generatedCount)
                    .totalTokenCount(generation.getPromptTokenCount() + generatedCount).build();
        }

        protected static VlmStopSequence matchedStop(VlmProtocolPlan plan, GenerationResult generation) {
            if (plan == null || generation == null || generation.getTokenIds() == null) return null;
            int[] generated = generation.getTokenIds();
            for (VlmStopSequence stop : plan.getStops()) {
                int[] sequence = stop.getTokenIds();
                if (sequence == null || sequence.length > generated.length) continue;
                int start = generated.length - sequence.length;
                boolean matches = true;
                for (int i = 0; i < sequence.length; i++) {
                    if (generated[start + i] != sequence[i]) { matches = false; break; }
                }
                if (matches) return stop;
            }
            return null;
        }

        protected static Map<String, Object> stopMetadata(VlmStopSequence stop) {
            if (stop == null) return Collections.emptyMap();
            Map<String, Object> metadata = new LinkedHashMap<>();
            metadata.put("matchedStopId", stop.getId());
            metadata.put("matchedStopKind", stop.getKind().name());
            metadata.put("matchedStopRetention", stop.getRetention().name());
            metadata.put("matchedStopTokenCount", stop.getTokenIds().length);
            return Collections.unmodifiableMap(metadata);
        }

        protected static GenerationResult mergeRegionResults(List<GenerationResult> regions,
                                                             String prefix, String suffix) {
            List<Integer> ids = new ArrayList<>();
            StringBuilder text = new StringBuilder(prefix == null ? "" : prefix);
            int generated = 0, prompts = 0;
            long time = 0, firstToken = 0;
            GenerationResult.FinishReason finishReason = GenerationResult.FinishReason.EOS;
            if (regions != null) for (GenerationResult region : regions) {
                if (region == null) continue;
                String part = region.getText();
                if (part != null && !part.isBlank()) {
                    if (text.length() > 0 && prefix == null) text.append('\n');
                    text.append(part);
                }
                if (region.getTokenIds() != null) for (int id : region.getTokenIds()) ids.add(id);
                generated += region.getGeneratedTokenCount();
                prompts += region.getPromptTokenCount();
                time += region.getGenerationTimeMs();
                firstToken += region.getFirstTokenLatencyMs();
                finishReason = strongerFinishReason(finishReason, region.getFinishReason());
            }
            if (suffix != null) text.append(suffix);
            int[] tokenIds = ids.stream().mapToInt(Integer::intValue).toArray();
            return GenerationResult.builder().text(text.toString()).tokenIds(tokenIds)
                    .generatedTokenCount(generated).promptTokenCount(prompts)
                    .totalTokenCount(generated + prompts).finishReason(finishReason)
                    .firstTokenLatencyMs(firstToken).generationTimeMs(time)
                    .tokensPerSecond(time > 0 ? generated * 1000.0 / time : 0.0).build();
        }

        private static GenerationResult.FinishReason strongerFinishReason(
                GenerationResult.FinishReason current, GenerationResult.FinishReason candidate) {
            if (candidate == null) return current;
            return finishPriority(candidate) > finishPriority(current) ? candidate : current;
        }

        private static int finishPriority(GenerationResult.FinishReason reason) {
            switch (reason) {
                case ERROR: return 6;
                case CANCELLED: return 5;
                case MAX_TOKENS: return 4;
                case REPETITION: return 3;
                case STOP_SEQUENCE: return 2;
                case EOS:
                default: return 1;
            }
        }

        private static boolean tagsBalanced(String text) {
            java.util.regex.Matcher matcher = java.util.regex.Pattern
                    .compile("<(/?)([A-Za-z_][A-Za-z0-9_-]*)([^>]*)>").matcher(text);
            List<String> stack = new ArrayList<>();
            boolean sawTag = false;
            while (matcher.find()) {
                sawTag = true;
                String name = matcher.group(2);
                if (matcher.group(3) != null && matcher.group(3).trim().endsWith("/")) continue;
                if (matcher.group(1).isEmpty()) stack.add(name);
                else {
                    if (stack.isEmpty() || !stack.remove(stack.size() - 1).equals(name)) return false;
                }
            }
            return sawTag && stack.isEmpty();
        }
    }

    private static class DocTagsProvider implements VlmOutputProtocolProvider {
        @Override public String providerId() { return "builtin.doctags"; }
        @Override public VlmOutputProtocol bind(String protocolId, VlmProtocolDefinition.Protocol definition) {
            return new DocTagsProtocol(protocolId, definition);
        }
    }

    private static final class DocTagsProtocol extends BoundProtocol {
        private final DocTagsParser parser = new DocTagsParser();

        private DocTagsProtocol(String protocolId, VlmProtocolDefinition.Protocol definition) {
            super(protocolId, definition, Mode.TAGGED);
        }

        @Override
        public VlmProtocolOutput process(VlmProtocolRequest request, VlmProtocolPlan plan,
                                         GenerationResult generation, Tokenizer tokenizer) {
            String raw = protocolText(plan, generation, tokenizer);
            DocumentStructure document = parser.parse(raw);
            boolean complete = generation != null
                    && generation.getFinishReason() != GenerationResult.FinishReason.MAX_TOKENS
                    && parser.isComplete(raw);
            VlmCompletion assessment = VlmCompletion.builder()
                    .complete(complete)
                    .usable(complete || !definition.getCompletion().isRequired())
                    .diagnostic(complete ? "complete DocTags envelope" : "incomplete DocTags envelope")
                    .build();
            VlmRenderFormat render = request.getRenderFormat() == null
                    ? VlmRenderFormat.PLAIN_TEXT : request.getRenderFormat();
            String rendered;
            switch (render) {
                case RAW: rendered = raw; break;
                case MARKDOWN: rendered = parser.toMarkdown(document); break;
                case HTML: rendered = parser.toHtml(document); break;
                case PLAIN_TEXT: rendered = document.getFullText(); break;
                default: rendered = raw;
            }
            return VlmProtocolOutput.builder()
                    .protocolId(protocolId)
                    .nativeFormat("DOCTAGS")
                    .rawText(raw)
                    .renderedText(rendered)
                    .structured(document)
                    .completion(assessment)
                    .metadata(stopMetadata(matchedStop(plan, generation)))
                    .build();
        }

        @Override
        public GenerationResult mergeRegions(VlmProtocolRequest request, VlmProtocolPlan plan,
                                             List<GenerationResult> regions, Tokenizer tokenizer) {
            List<GenerationResult> bodies = new ArrayList<>();
            if (regions != null) for (GenerationResult region : regions) {
                if (region == null) continue;
                String raw = protocolText(plan, region, tokenizer).trim();
                if (raw.startsWith("<doctag>")) raw = raw.substring("<doctag>".length());
                if (raw.endsWith("</doctag>")) raw = raw.substring(0, raw.length() - "</doctag>".length());
                bodies.add(region.toBuilder().text(raw).build());
            }
            return mergeRegionResults(bodies, "<doctag>", "</doctag>");
        }
    }
}
