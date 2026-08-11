/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Dependency-free host CLI for the mainline SDX compile/cache API.
 *
 * <p>The preferred path is a target compiler executable. A prepared-artifact
 * adapter exists only to migrate existing Vulkan/Hexagon/vendor outputs into
 * the same immutable cache without keeping packaging logic in applications.</p>
 */
public final class SdxCompilerCli {
    private SdxCompilerCli() {
    }

    public static void main(String[] args) {
        try {
            int status = run(args, System.out, System.err);
            if (status != 0) {
                System.exit(status);
            }
        } catch (Throwable failure) {
            failure.printStackTrace(System.err);
            System.exit(1);
        }
    }

    static int run(String[] args, PrintStream output, PrintStream error)
            throws IOException {
        if (args.length == 0 || "--help".equals(args[0]) || "-h".equals(args[0])) {
            usage(output);
            return 0;
        }

        String command = args[0].toLowerCase(Locale.ROOT);
        if (!isKnownCommand(command)) {
            error.println("Unknown SDX compiler command: " + command);
            usage(error);
            return 2;
        }
        Arguments options = parseArguments(
                command, Arrays.copyOfRange(args, 1, args.length));
        SdxModelCache cache = options.has("cache")
                ? new SdxModelCache(Path.of(options.required("cache")))
                : SdxModelCache.defaultCache();

        switch (command) {
            case "compile":
                return compile(cache, options, output);
            case "package":
                return packageModel(cache, options, output);
            case "resolve":
                return resolve(cache, options, output);
            case "validate-quantization":
                return validateQuantization(options, output);
            case "hexagon-plan":
                return hexagonPlan(options, output);
            case "hexagon-finalize":
                return hexagonFinalize(options, output);
            case "hexagon-verify":
                return hexagonVerify(options, output);
            default:
                throw new IllegalStateException("unhandled command: " + command);
        }
    }

    private static boolean isKnownCommand(String command) {
        return "compile".equals(command)
                || "package".equals(command)
                || "resolve".equals(command)
                || "validate-quantization".equals(command)
                || "hexagon-plan".equals(command)
                || "hexagon-finalize".equals(command)
                || "hexagon-verify".equals(command);
    }

    private static Arguments parseArguments(String command, String[] args) {
        switch (command) {
            case "compile":
                return Arguments.parse(command, args, Set.of(
                        "cache", "source", "target", "compiler-id",
                        "compiler-version", "prepared-artifact", "compiler-command",
                        "compiler-arg", "compiler-fingerprint", "tokenizer",
                        "tokenizer-config",
                        "llm-config", "quantization-config", "model-id",
                        "cache-key-option", "package-output"),
                        Set.of("compiler-arg", "cache-key-option"), Set.of());
            case "package":
                return Arguments.parse(command, args,
                        Set.of("cache", "source", "targets", "output"),
                        Set.of(), Set.of());
            case "resolve":
                return Arguments.parse(command, args,
                        Set.of("cache", "source", "target"), Set.of(), Set.of());
            case "validate-quantization":
                return Arguments.parse(command, args,
                        Set.of("config"), Set.of(), Set.of());
            case "hexagon-plan":
                return Arguments.parse(command, args, Set.of(
                        "segments-json", "soc", "model-id", "output",
                        "include-noncapturable", "allow-unstable", "allow-empty"),
                        Set.of(), Set.of(
                                "include-noncapturable", "allow-unstable",
                                "allow-empty"));
            case "hexagon-finalize":
            case "hexagon-verify":
                return Arguments.parse(command, args,
                        Set.of("request", "kernel-dir"), Set.of(), Set.of());
            default:
                throw new IllegalArgumentException("Unknown command: " + command);
        }
    }

    private static int compile(
            SdxModelCache cache, Arguments args, PrintStream output) throws IOException {
        Path source = Path.of(args.required("source"));
        SdxTargetProfile target =
                SdxTargetProfile.fromId(args.required("target"));
        String compilerId = args.required("compiler-id");
        String compilerVersion = args.required("compiler-version");

        SdxModelCompiler.TargetCompiler compiler;
        if (args.has("prepared-artifact")) {
            if (args.has("compiler-command")) {
                throw new IllegalArgumentException(
                        "Use either --prepared-artifact or --compiler-command");
            }
            compiler = SdxModelCompiler.preparedArtifact(
                    Path.of(args.required("prepared-artifact")),
                    compilerId,
                    compilerVersion);
        } else {
            List<String> command = new ArrayList<>();
            command.add(args.required("compiler-command"));
            command.addAll(args.values("compiler-arg"));
            compiler = SdxModelCompiler.externalCommand(
                    command,
                    compilerId,
                    compilerVersion,
                    args.required("compiler-fingerprint"));
        }

        SdxModelCompiler.CompileOptions.Builder compileOptions =
                SdxModelCompiler.CompileOptions.builder();
        args.optionalPath("tokenizer").ifPresent(compileOptions::tokenizer);
        args.optionalPath("tokenizer-config").ifPresent(compileOptions::tokenizerConfig);
        args.optionalPath("llm-config")
                .ifPresent(compileOptions::textGenerationConfig);
        args.optionalPath("quantization-config")
                .ifPresent(compileOptions::quantizationConfig);
        if (args.has("model-id")) {
            compileOptions.modelId(args.required("model-id"));
        }
        for (String option : args.values("cache-key-option")) {
            int separator = option.indexOf('=');
            if (separator <= 0) {
                throw new IllegalArgumentException(
                        "--cache-key-option must be key=value");
            }
            compileOptions.cacheKeyProperty(
                    option.substring(0, separator),
                    option.substring(separator + 1));
        }

        SdxCompiledModel compiled = new SdxModelCompiler(cache).compile(
                source, target, compiler, compileOptions.build());
        printResult(compiled, output);

        if (args.has("package-output")) {
            cache.packageCompiledSdz(
                    source,
                    java.util.Collections.singletonList(target),
                    Path.of(args.required("package-output")));
            output.println("packagedSdz=" + Path.of(
                    args.required("package-output")).toAbsolutePath().normalize());
        }
        return 0;
    }

    private static int packageModel(
            SdxModelCache cache, Arguments args, PrintStream output) throws IOException {
        Path source = Path.of(args.required("source"));
        String[] ids = args.required("targets").split(",");
        List<SdxTargetProfile> targets = new ArrayList<>();
        for (String id : ids) {
            if (!id.trim().isEmpty()) {
                targets.add(SdxTargetProfile.fromId(id.trim()));
            }
        }
        if (targets.isEmpty()) {
            throw new IllegalArgumentException("--targets contains no target profiles");
        }
        Path packaged = Path.of(args.required("output"));
        cache.packageCompiledSdz(source, targets, packaged);
        output.println("packagedSdz=" + packaged.toAbsolutePath().normalize());
        output.println("sourceSha256=" + cache.identify(source).sha256());
        return 0;
    }

    private static int resolve(
            SdxModelCache cache, Arguments args, PrintStream output) throws IOException {
        SdxCompiledModel model = cache.resolve(
                Path.of(args.required("source")),
                SdxTargetProfile.fromId(args.required("target")));
        printResult(model, output);
        return 0;
    }

    private static int validateQuantization(
            Arguments args, PrintStream output) throws IOException {
        SdxQuantizationContract contract = SdxQuantizationContract.load(
                Path.of(args.required("config")));
        output.println(contract.summaryJson());
        return 0;
    }

    private static int hexagonPlan(Arguments args, PrintStream output) throws IOException {
        int count = HexagonAot.plan(
                Path.of(args.required("segments-json")),
                args.required("soc"),
                args.has("model-id") ? args.required("model-id") : null,
                Path.of(args.required("output")),
                args.flag("include-noncapturable"),
                args.flag("allow-unstable"),
                args.flag("allow-empty"));
        output.println("wrote " + count + " Hexagon AOT requests to "
                + args.required("output"));
        return 0;
    }

    private static int hexagonFinalize(Arguments args, PrintStream output)
            throws IOException {
        int count = HexagonAot.finalizeArtifacts(
                Path.of(args.required("request")),
                Path.of(args.required("kernel-dir")));
        output.println("finalized " + count + " Hexagon AOT artifacts under "
                + args.required("kernel-dir"));
        return 0;
    }

    private static int hexagonVerify(Arguments args, PrintStream output)
            throws IOException {
        int count = HexagonAot.verify(
                Path.of(args.required("request")),
                Path.of(args.required("kernel-dir")));
        output.println("verified " + count + " Hexagon AOT artifacts");
        return 0;
    }

    private static void printResult(SdxCompiledModel model, PrintStream output) {
        output.println("sourceSha256=" + model.sourceIdentity().sha256());
        output.println("target=" + model.target().id());
        output.println("compileKey=" + model.compileKey());
        output.println("compiler=" + model.compilerId() + "@" + model.compilerVersion());
        output.println("runtimePath=" + model.runtimeModelPath());
        model.tokenizerPath().ifPresent(path -> output.println("tokenizerPath=" + path));
        model.tokenizerConfigPath().ifPresent(
                path -> output.println("tokenizerConfigPath=" + path));
        model.textGenerationConfigPath().ifPresent(
                path -> output.println("textGenerationConfigPath=" + path));
        model.quantizationConfigPath().ifPresent(
                path -> output.println("quantizationConfigPath=" + path));
    }

    private static void usage(PrintStream output) {
        output.println("Usage:");
        output.println("  SdxCompilerCli compile --source model.sdz --target <profile>");
        output.println("      --cache <dir> --compiler-id <id> --compiler-version <version>");
        output.println("      (--compiler-command <exe> --compiler-fingerprint <digest>");
        output.println("       [--compiler-arg <arg> ...] | --prepared-artifact <path>)");
        output.println("      [--tokenizer <file>] [--tokenizer-config <file>]");
        output.println("      [--llm-config <file>]");
        output.println("      [--quantization-config <file>] [--cache-key-option k=v]");
        output.println("      [--package-output compiled-model.sdz]");
        output.println("  SdxCompilerCli package --source model.sdz --cache <dir>");
        output.println("      --targets <csv> --output compiled-model.sdz");
        output.println("  SdxCompilerCli resolve --source model.sdz --cache <dir>");
        output.println("      --target <profile>");
        output.println("  SdxCompilerCli validate-quantization --config <file>");
        output.println("  SdxCompilerCli hexagon-plan --segments-json <file> --soc <soc>");
        output.println("      [--model-id <id>] --output <request.json>");
        output.println("      [--include-noncapturable] [--allow-unstable] [--allow-empty]");
        output.println("  SdxCompilerCli hexagon-finalize --request <request.json>");
        output.println("      --kernel-dir <compiled-kernels>");
        output.println("  SdxCompilerCli hexagon-verify --request <request.json>");
        output.println("      --kernel-dir <compiled-kernels>");
        output.println();
        output.println("Known profiles:");
        for (SdxTargetProfile profile : SdxTargetProfile.values()) {
            output.println("  " + profile.id());
        }
    }

    private static final class Arguments {
        private final Map<String, List<String>> values;

        private Arguments(Map<String, List<String>> values) {
            this.values = values;
        }

        private static Arguments parse(
                String command,
                String[] args,
                Set<String> allowed,
                Set<String> repeatable,
                Set<String> flags) {
            Map<String, List<String>> values = new LinkedHashMap<>();
            for (int i = 0; i < args.length; i++) {
                String token = args[i];
                if (!token.startsWith("--") || token.length() <= 2) {
                    throw new IllegalArgumentException("Expected --option, got: " + token);
                }
                String key = token.substring(2);
                if (!allowed.contains(key)) {
                    throw new IllegalArgumentException(
                            "Unknown option for " + command + ": --" + key);
                }
                if (values.containsKey(key) && !repeatable.contains(key)) {
                    throw new IllegalArgumentException(
                            "Option may be specified only once: --" + key);
                }
                if (flags.contains(key)) {
                    if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                        throw new IllegalArgumentException(
                                "--" + key + " does not take a value");
                    }
                    values.put(key, new ArrayList<>());
                    continue;
                }
                if (i + 1 >= args.length || args[i + 1].startsWith("--")) {
                    throw new IllegalArgumentException("Missing value for --" + key);
                }
                values.computeIfAbsent(key, ignored -> new ArrayList<>())
                        .add(args[++i]);
            }
            return new Arguments(values);
        }

        private boolean has(String key) {
            return values.containsKey(key);
        }

        private boolean flag(String key) {
            List<String> found = values.get(key);
            if (found == null) {
                return false;
            }
            if (!found.isEmpty()) {
                throw new IllegalArgumentException("--" + key + " does not take a value");
            }
            return true;
        }

        private String required(String key) {
            List<String> found = values.get(key);
            if (found == null || found.isEmpty() || found.get(found.size() - 1).isEmpty()) {
                throw new IllegalArgumentException("Missing required --" + key);
            }
            return found.get(found.size() - 1);
        }

        private List<String> values(String key) {
            return values.getOrDefault(key, java.util.Collections.emptyList());
        }

        private java.util.Optional<Path> optionalPath(String key) {
            return has(key)
                    ? java.util.Optional.of(Path.of(required(key)))
                    : java.util.Optional.empty();
        }
    }
}
