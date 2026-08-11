/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.eclipse.deeplearning4j.sdx.aot;

import org.eclipse.deeplearning4j.llm.generation.GenerationResult;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * {@code sdx-llm} — the SDX SDK's ahead-of-time compiled LLM tool. A single native
 * executable (GraalVM native-image, no JVM) exposing the Java LLM stack to SDK users:
 *
 * <pre>
 *   sdx-llm generate --model model.gguf --prompt "..." [--tokenizer dir] [--max-new-tokens N]
 *                    [--temperature T --top-k K --top-p P --seed S | --greedy]
 *                    [--chat [--system "..."]] [--stats]
 *   sdx-llm chat     --model model.gguf --request-file request.json [generation options]
 *   sdx-llm import   --model model.gguf --output model.sdz
 *   sdx-llm tokenize --model model.gguf --text "..." [--no-special]
 *   sdx-llm info     --model model.gguf
 * </pre>
 *
 * {@code import} converts GGUF to SDZ so the JVM-free C runtime (libsdx / dsp_runtime_c.h)
 * can execute the graph directly; {@code generate} runs the full pipeline in-process.
 */
public final class SdxLlmCli {

    private SdxLlmCli() {
    }

    public static void main(String[] args) {
        // Point JavaCPP at the side-loaded natives (SDK lib/ next to the binary)
        // before anything can initialize ND4J or the tokenizers (ADR 0109).
        SdxNativeLibs.bootstrap();
        try {
            System.exit(run(args));
        } catch (Exception e) {
            System.err.println("sdx-llm: error: " + e.getMessage());
            if (Boolean.getBoolean("sdx.llm.verbose")) {
                e.printStackTrace();
            }
            System.exit(1);
        }
    }

    static int run(String[] args) throws Exception {
        if (args.length == 0 || isHelp(args[0])) {
            printUsage();
            return args.length == 0 ? 2 : 0;
        }
        String command = args[0];
        Map<String, String> opts = parseOptions(args);

        switch (command) {
            case "generate":
                return generate(opts);
            case "chat":
                return chat(opts);
            case "import":
                return importModel(opts);
            case "tokenize":
                return tokenize(opts);
            case "info":
                return info(opts);
            case "vlm":
                return vlm(opts);
            case "transcribe":
                return transcribe(opts);
            case "version":
                System.out.println("sdx-llm " + version());
                return 0;
            default:
                System.err.println("sdx-llm: unknown command: " + command);
                printUsage();
                return 2;
        }
    }

    private static int vlm(Map<String, String> opts) throws Exception {
        String model = required(opts, "model");
        String input = required(opts, "input");
        StringBuilder json = new StringBuilder("{");
        json.append("\"maxNewTokens\":").append(opts.containsKey("max-new-tokens")
                ? Integer.parseInt(opts.get("max-new-tokens")) : 1024);
        if (opts.containsKey("prompt")) {
            json.append(",\"prompt\":").append(quoteJson(opts.get("prompt")));
        }
        if (opts.containsKey("format")) {
            json.append(",\"format\":").append(quoteJson(opts.get("format")));
        }
        json.append("}");
        System.out.println(SdxVlm.extract(model, opts.get("tokenizer"), input, json.toString()));
        return 0;
    }

    private static int transcribe(Map<String, String> opts) throws Exception {
        String model = required(opts, "model");
        String audio = required(opts, "audio");
        StringBuilder json = new StringBuilder("{");
        if (opts.containsKey("language")) {
            json.append("\"language\":").append(quoteJson(opts.get("language")));
        }
        if (opts.containsKey("max-new-tokens")) {
            if (json.length() > 1) json.append(",");
            json.append("\"maxNewTokens\":").append(Integer.parseInt(opts.get("max-new-tokens")));
        }
        json.append("}");
        System.out.println(SdxAudio.transcribe(model, audio, json.toString()));
        return 0;
    }

    private static String quoteJson(String value) {
        return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    private static int generate(Map<String, String> opts) throws Exception {
        String model = required(opts, "model");
        String prompt = opts.get("prompt");
        if (prompt == null && opts.containsKey("prompt-file")) {
            prompt = new String(Files.readAllBytes(Paths.get(opts.get("prompt-file"))), StandardCharsets.UTF_8);
        }
        if (prompt == null) {
            throw new IllegalArgumentException("generate requires --prompt or --prompt-file");
        }

        String loadOptions = buildOptionsJson(opts);
        try (SdxLlmCore core = SdxLlmCore.load(model, opts.get("tokenizer"), loadOptions)) {
            String effectivePrompt = prompt;
            if (opts.containsKey("chat")) {
                Tokenizer tokenizer = core.tokenizer();
                List<ChatTemplate.Message> messages = new ArrayList<>();
                if (opts.containsKey("system")) {
                    messages.add(ChatTemplate.Message.system(opts.get("system")));
                }
                messages.add(ChatTemplate.Message.user(prompt));
                effectivePrompt = tokenizer.applyChatTemplate(messages, true);
            }

            GenerationResult result = core.generate(effectivePrompt, null);
            System.out.println(result.getText());
            if (opts.containsKey("stats")) {
                System.err.println(core.lastResultJson());
            }
        }
        return 0;
    }

    private static int chat(Map<String, String> opts) throws Exception {
        String model = required(opts, "model");
        String requestJson = opts.get("request");
        if (requestJson == null && opts.containsKey("request-file")) {
            requestJson = new String(
                    Files.readAllBytes(Paths.get(opts.get("request-file"))),
                    StandardCharsets.UTF_8);
        }
        if (requestJson == null || requestJson.isBlank()) {
            throw new IllegalArgumentException(
                    "chat requires --request or --request-file");
        }
        String optionsJson = buildOptionsJson(opts);
        try (SdxLlmCore core = SdxLlmCore.load(
                model, opts.get("tokenizer"), optionsJson)) {
            System.out.println(core.generateChat(requestJson, optionsJson));
        }
        return 0;
    }

    private static int importModel(Map<String, String> opts) throws Exception {
        String model = required(opts, "model");
        String output = required(opts, "output");
        if (!output.toLowerCase().endsWith(".sdz")) {
            throw new IllegalArgumentException("import output must end with .sdz (got " + output + ")");
        }
        SameDiff sd = SdxLlmCore.loadDecoder(model);
        Map<String, String> metadata = new HashMap<>();
        metadata.put("sdx.aot.source", new File(model).getName());
        metadata.put("sdx.aot.tool", "sdx-llm " + version());
        SDZSerializer.save(sd, new File(output), false, metadata);
        System.out.println("Imported " + model + " -> " + output);
        return 0;
    }

    private static int tokenize(Map<String, String> opts) throws Exception {
        String model = opts.get("model");
        String tokenizer = opts.get("tokenizer");
        if (model == null && tokenizer == null) {
            throw new IllegalArgumentException("tokenize requires --model or --tokenizer");
        }
        String text = required(opts, "text");
        Tokenizer tok = SdxLlmCore.resolveTokenizer(model != null ? model : tokenizer, tokenizer);
        int[] ids = tok.encode(text, !opts.containsKey("no-special")).getIds();
        System.out.println(Arrays.toString(ids));
        System.err.println("count=" + ids.length + " decoded=" + tok.decode(ids, true));
        return 0;
    }

    private static int info(Map<String, String> opts) throws Exception {
        String model = required(opts, "model");
        try (SdxLlmCore core = SdxLlmCore.load(model, opts.get("tokenizer"),
                "{\"maxNewTokens\":1}")) {
            System.out.println(core.infoJson());
        }
        return 0;
    }

    /** Translate CLI sampling flags into the shared options JSON understood by SdxLlmCore. */
    private static String buildOptionsJson(Map<String, String> opts) {
        StringBuilder sampling = new StringBuilder();
        if (opts.containsKey("greedy")) {
            sampling.append("{\"preset\":\"greedy\"}");
        } else {
            List<String> fields = new ArrayList<>();
            if (opts.containsKey("temperature")) fields.add("\"temperature\":" + Double.parseDouble(opts.get("temperature")));
            if (opts.containsKey("top-k")) fields.add("\"topK\":" + Integer.parseInt(opts.get("top-k")));
            if (opts.containsKey("top-p")) fields.add("\"topP\":" + Double.parseDouble(opts.get("top-p")));
            if (opts.containsKey("repetition-penalty")) fields.add("\"repetitionPenalty\":" + Double.parseDouble(opts.get("repetition-penalty")));
            if (opts.containsKey("seed")) fields.add("\"seed\":" + Long.parseLong(opts.get("seed")));
            if (!fields.isEmpty()) {
                sampling.append("{").append(String.join(",", fields)).append("}");
            }
        }
        StringBuilder json = new StringBuilder("{");
        json.append("\"maxNewTokens\":").append(opts.containsKey("max-new-tokens")
                ? Integer.parseInt(opts.get("max-new-tokens")) : 128);
        if (opts.containsKey("no-graph-optimizer")) {
            json.append(",\"graphOptimizer\":false");
        }
        if (sampling.length() > 0) {
            json.append(",\"sampling\":").append(sampling);
        }
        json.append("}");
        return json.toString();
    }

    /** `--key value` and boolean `--flag` parsing; values never start with `--`. */
    private static Map<String, String> parseOptions(String[] args) {
        Map<String, String> opts = new HashMap<>();
        for (int i = 1; i < args.length; i++) {
            String arg = args[i];
            if (!arg.startsWith("--")) {
                throw new IllegalArgumentException("Unexpected argument: " + arg);
            }
            String key = arg.substring(2);
            if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                opts.put(key, args[++i]);
            } else {
                opts.put(key, "");
            }
        }
        return opts;
    }

    private static String required(Map<String, String> opts, String key) {
        String value = opts.get(key);
        if (value == null || value.isEmpty()) {
            throw new IllegalArgumentException("Missing required option: --" + key);
        }
        return value;
    }

    private static boolean isHelp(String arg) {
        return "--help".equals(arg) || "-h".equals(arg) || "help".equals(arg);
    }

    static String version() {
        String v = SdxLlmCli.class.getPackage() != null
                ? SdxLlmCli.class.getPackage().getImplementationVersion() : null;
        return v != null ? v : "dev";
    }

    private static void printUsage() {
        System.out.println(String.join(System.lineSeparator(),
                "sdx-llm — SDX SDK LLM tool (AOT-compiled, no JVM required)",
                "",
                "Commands:",
                "  generate   --model <path> --prompt <text> [options]   Run text generation",
                "  chat       --model <path> --request-file <json>       Run structured model-owned chat",
                "  import     --model <in.gguf> --output <out.sdz>       Convert GGUF to SDZ for the C runtime",
                "  tokenize   --model <path>|--tokenizer <path> --text <text> [--no-special]",
                "  info       --model <path> [--tokenizer <path>]        Print model/tokenizer summary JSON",
                "  vlm        --model <path> --input <img|pdf> [--tokenizer <path>] [--prompt <text>]",
                "             [--format doctags|markdown|text] [--max-new-tokens <n>]",
                "  transcribe --model <path> --audio <wav> [--language <code>] [--max-new-tokens <n>]",
                "  version",
                "",
                "generate options:",
                "  --tokenizer <path>          tokenizer.json file or directory (default: model dir)",
                "  --prompt-file <path>        read the prompt from a file",
                "  --max-new-tokens <n>        default 128",
                "  --greedy                    deterministic decoding (default)",
                "  --temperature <t> --top-k <k> --top-p <p> --repetition-penalty <r> --seed <s>",
                "  --chat [--system <text>]    apply the tokenizer's chat template",
                "  --no-graph-optimizer        disable graph optimizer passes",
                "  --stats                     print generation stats JSON to stderr"));
    }
}
