/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.llm.tokenizer;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * CLIP tokenizer implementation.
 *
 * This tokenizer is used by CLIP, OpenCLIP, and related vision-language models.
 * It uses byte-level BPE tokenization similar to GPT-2.
 *
 * <p>Usage:</p>
 * <pre>{@code
 * CLIPTokenizer tokenizer = CLIPTokenizer.fromFiles(
 *     new File("vocab.json"),
 *     new File("merges.txt")
 * );
 * Encoding encoding = tokenizer.encode("A photo of a cat");
 * }</pre>
 *
 * @author Eclipse Deeplearning4j Contributors
 */
public class CLIPTokenizer implements Tokenizer {

    private static final Pattern PAT = Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
            Pattern.UNICODE_CHARACTER_CLASS
    );

    private final Map<String, Integer> encoder;
    private final Map<Integer, String> decoder;
    private final Map<Pair<String, String>, Integer> bpeRanks;
    private final Map<Integer, Character> byteEncoder;
    private final Map<Character, Integer> byteDecoder;
    private final Map<String, String> cache;

    private int bosTokenId;
    private int eosTokenId;
    private int padTokenId;
    private int unkTokenId;

    private int contextLength = 77;

    private CLIPTokenizer(Map<String, Integer> encoder, Map<Pair<String, String>, Integer> bpeRanks) {
        this.encoder = encoder;
        this.decoder = new HashMap<>();
        for (Map.Entry<String, Integer> entry : encoder.entrySet()) {
            decoder.put(entry.getValue(), entry.getKey());
        }
        this.bpeRanks = bpeRanks;
        this.byteEncoder = bytesToUnicode();
        this.byteDecoder = new HashMap<>();
        for (Map.Entry<Integer, Character> entry : byteEncoder.entrySet()) {
            byteDecoder.put(entry.getValue(), entry.getKey());
        }
        this.cache = new HashMap<>();

        // Find special tokens
        this.bosTokenId = encoder.getOrDefault("<|startoftext|>", 49406);
        this.eosTokenId = encoder.getOrDefault("<|endoftext|>", 49407);
        this.padTokenId = 0;
        this.unkTokenId = encoder.getOrDefault("<|endoftext|>", 49407);
    }

    /**
     * Create a CLIP tokenizer from vocab and merges files.
     *
     * @param vocabFile the vocab.json file
     * @param mergesFile the merges.txt file
     * @return the tokenizer
     * @throws IOException if loading fails
     */
    public static CLIPTokenizer fromFiles(File vocabFile, File mergesFile) throws IOException {
        // Load vocabulary
        Map<String, Integer> encoder = loadVocab(vocabFile);

        // Load BPE merges
        Map<Pair<String, String>, Integer> bpeRanks = loadMerges(mergesFile);

        return new CLIPTokenizer(encoder, bpeRanks);
    }

    /**
     * Create a CLIP tokenizer from a directory containing vocab.json and merges.txt.
     *
     * @param directory the directory
     * @return the tokenizer
     * @throws IOException if loading fails
     */
    public static CLIPTokenizer fromDirectory(File directory) throws IOException {
        File vocabFile = new File(directory, "vocab.json");
        File mergesFile = new File(directory, "merges.txt");
        return fromFiles(vocabFile, mergesFile);
    }

    private static Map<String, Integer> loadVocab(File file) throws IOException {
        Map<String, Integer> vocab = new LinkedHashMap<>();
        String content = new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);

        // Simple JSON parsing
        content = content.trim();
        if (content.startsWith("{") && content.endsWith("}")) {
            content = content.substring(1, content.length() - 1);
        }

        Pattern pattern = Pattern.compile("\"([^\"\\\\]|\\\\.)*\"\\s*:\\s*(\\d+)");
        Matcher matcher = pattern.matcher(content);
        while (matcher.find()) {
            String token = matcher.group(0);
            int colonIdx = token.lastIndexOf(':');
            String key = token.substring(1, colonIdx).trim();
            key = key.substring(0, key.length() - 1); // Remove trailing quote
            key = unescapeJson(key);
            int value = Integer.parseInt(token.substring(colonIdx + 1).trim());
            vocab.put(key, value);
        }

        return vocab;
    }

    private static String unescapeJson(String s) {
        return s.replace("\\\"", "\"")
                .replace("\\\\", "\\")
                .replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace("\\t", "\t");
    }

    private static Map<Pair<String, String>, Integer> loadMerges(File file) throws IOException {
        Map<Pair<String, String>, Integer> merges = new LinkedHashMap<>();
        List<String> lines = Files.readAllLines(file.toPath(), StandardCharsets.UTF_8);

        int rank = 0;
        for (String line : lines) {
            if (line.startsWith("#")) continue;
            String[] parts = line.split(" ");
            if (parts.length == 2) {
                merges.put(new Pair<>(parts[0], parts[1]), rank++);
            }
        }

        return merges;
    }

    private static Map<Integer, Character> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>();
        for (int i = '!'; i <= '~'; i++) bs.add(i);
        for (int i = '¡'; i <= '¬'; i++) bs.add(i);
        for (int i = '®'; i <= 'ÿ'; i++) bs.add(i);

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n++;
            }
        }

        Map<Integer, Character> result = new HashMap<>();
        for (int i = 0; i < bs.size(); i++) {
            result.put(bs.get(i), (char) (int) cs.get(i));
        }
        return result;
    }

    private List<String> getPairs(List<String> word) {
        List<String> pairs = new ArrayList<>();
        for (int i = 0; i < word.size() - 1; i++) {
            pairs.add(word.get(i) + " " + word.get(i + 1));
        }
        return pairs;
    }

    private String bpe(String token) {
        if (cache.containsKey(token)) {
            return cache.get(token);
        }

        List<String> word = new ArrayList<>();
        for (int i = 0; i < token.length() - 1; i++) {
            word.add(String.valueOf(token.charAt(i)));
        }
        word.add(token.charAt(token.length() - 1) + "</w>");

        while (word.size() > 1) {
            Pair<String, String> minPair = null;
            int minRank = Integer.MAX_VALUE;

            for (int i = 0; i < word.size() - 1; i++) {
                Pair<String, String> pair = new Pair<>(word.get(i), word.get(i + 1));
                Integer rank = bpeRanks.get(pair);
                if (rank != null && rank < minRank) {
                    minRank = rank;
                    minPair = pair;
                }
            }

            if (minPair == null) break;

            List<String> newWord = new ArrayList<>();
            int i = 0;
            while (i < word.size()) {
                if (i < word.size() - 1 &&
                        word.get(i).equals(minPair.first) &&
                        word.get(i + 1).equals(minPair.second)) {
                    newWord.add(minPair.first + minPair.second);
                    i += 2;
                } else {
                    newWord.add(word.get(i));
                    i++;
                }
            }
            word = newWord;
        }

        String result = String.join(" ", word);
        cache.put(token, result);
        return result;
    }

    @Override
    public Encoding encode(String text, boolean addSpecialTokens) {
        List<Integer> bpeTokens = new ArrayList<>();
        List<String> tokens = new ArrayList<>();

        if (addSpecialTokens) {
            bpeTokens.add(bosTokenId);
            tokens.add("<|startoftext|>");
        }

        text = text.toLowerCase().trim();

        Matcher matcher = PAT.matcher(text);
        while (matcher.find()) {
            String word = matcher.group();
            StringBuilder encoded = new StringBuilder();
            for (byte b : word.getBytes(StandardCharsets.UTF_8)) {
                encoded.append(byteEncoder.get(b & 0xFF));
            }

            String bpeWord = bpe(encoded.toString());
            for (String bpeToken : bpeWord.split(" ")) {
                Integer id = encoder.get(bpeToken);
                if (id != null) {
                    bpeTokens.add(id);
                    tokens.add(bpeToken);
                }
            }
        }

        if (addSpecialTokens) {
            bpeTokens.add(eosTokenId);
            tokens.add("<|endoftext|>");
        }

        // Pad or truncate to context length
        int[] ids = new int[contextLength];
        int[] attentionMask = new int[contextLength];
        Arrays.fill(ids, padTokenId);

        int len = Math.min(bpeTokens.size(), contextLength);
        for (int i = 0; i < len; i++) {
            ids[i] = bpeTokens.get(i);
            attentionMask[i] = 1;
        }

        return Encoding.builder()
                .ids(ids)
                .tokens(tokens.toArray(new String[0]))
                .attentionMask(attentionMask)
                .build();
    }

    @Override
    public List<Encoding> encodeBatch(List<String> texts, boolean addSpecialTokens) {
        List<Encoding> results = new ArrayList<>();
        for (String text : texts) {
            results.add(encode(text, addSpecialTokens));
        }
        return results;
    }

    @Override
    public String decode(int[] ids, boolean skipSpecialTokens) {
        StringBuilder result = new StringBuilder();
        for (int id : ids) {
            if (id == padTokenId) continue;
            if (skipSpecialTokens && (id == bosTokenId || id == eosTokenId)) {
                continue;
            }
            String token = decoder.get(id);
            if (token != null) {
                result.append(token);
            }
        }

        // Convert byte-level BPE back to text
        String text = result.toString();
        text = text.replace("</w>", " ");

        StringBuilder decoded = new StringBuilder();
        for (char c : text.toCharArray()) {
            Integer b = byteDecoder.get(c);
            if (b != null) {
                decoded.append((char) (int) b);
            }
        }

        return decoded.toString().trim();
    }

    @Override
    public List<String> decodeBatch(List<int[]> idsBatch, boolean skipSpecialTokens) {
        List<String> results = new ArrayList<>();
        for (int[] ids : idsBatch) {
            results.add(decode(ids, skipSpecialTokens));
        }
        return results;
    }

    @Override
    public int getVocabSize() {
        return encoder.size();
    }

    @Override
    public Integer getTokenId(String token) {
        return encoder.get(token);
    }

    @Override
    public String getToken(int id) {
        return decoder.get(id);
    }

    @Override
    public Map<String, Integer> getVocab() {
        return Collections.unmodifiableMap(encoder);
    }

    @Override
    public int getPadTokenId() {
        return padTokenId;
    }

    @Override
    public int getBosTokenId() {
        return bosTokenId;
    }

    @Override
    public int getEosTokenId() {
        return eosTokenId;
    }

    @Override
    public int getUnkTokenId() {
        return unkTokenId;
    }

    @Override
    public boolean isValid() {
        return encoder != null && !encoder.isEmpty();
    }

    /**
     * Set the context length (max sequence length).
     *
     * @param contextLength the context length
     */
    public void setContextLength(int contextLength) {
        this.contextLength = contextLength;
    }

    /**
     * Get the context length.
     *
     * @return the context length
     */
    public int getContextLength() {
        return contextLength;
    }

    @Override
    public void close() {
        cache.clear();
    }

    private static class Pair<A, B> {
        final A first;
        final B second;

        Pair(A first, B second) {
            this.first = first;
            this.second = second;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Pair<?, ?> pair = (Pair<?, ?>) o;
            return Objects.equals(first, pair.first) && Objects.equals(second, pair.second);
        }

        @Override
        public int hashCode() {
            return Objects.hash(first, second);
        }
    }
}
