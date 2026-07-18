package org.eclipse.deeplearning4j.llm.finetune;

import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Reproducibility metadata for one fine-tuning run. */
public class FineTuneManifest {
    private String corpusFingerprint;
    private String splitSalt;
    private Map<String, Integer> splitCounts = new LinkedHashMap<>();
    private String tokenizerIdentity;
    private int sequenceLength;
    private Map<String, Object> studentConfig = new LinkedHashMap<>();
    private Map<String, Object> provenance = new LinkedHashMap<>();

    public static String fingerprint(List<GeneratedTrainingExample> examples) {
        try {
            ObjectMapper mapper = mapper();
            List<GeneratedTrainingExample> ordered = new ArrayList<>(examples);
            ordered.sort(Comparator.comparing(GeneratedTrainingExample::getId));
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            for (GeneratedTrainingExample example : ordered) {
                example.validate();
                digest.update(mapper.writeValueAsBytes(example));
                digest.update((byte) '\n');
            }
            StringBuilder out = new StringBuilder();
            for (byte b : digest.digest()) out.append(String.format("%02x", b));
            return out.toString();
        } catch (Exception e) {
            throw new IllegalStateException("Unable to fingerprint corpus", e);
        }
    }

    public String toJson() {
        try { return mapper().writeValueAsString(this); }
        catch (Exception e) { throw new IllegalStateException("Unable to serialize manifest", e); }
    }

    private static ObjectMapper mapper() {
        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        mapper.configure(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS, true);
        return mapper;
    }

    public String getCorpusFingerprint() { return corpusFingerprint; }
    public void setCorpusFingerprint(String value) { corpusFingerprint = value; }
    public String getSplitSalt() { return splitSalt; }
    public void setSplitSalt(String value) { splitSalt = value; }
    public Map<String, Integer> getSplitCounts() { return splitCounts; }
    public void setSplitCounts(Map<String, Integer> value) { splitCounts = value; }
    public String getTokenizerIdentity() { return tokenizerIdentity; }
    public void setTokenizerIdentity(String value) { tokenizerIdentity = value; }
    public int getSequenceLength() { return sequenceLength; }
    public void setSequenceLength(int value) { sequenceLength = value; }
    public Map<String, Object> getStudentConfig() { return studentConfig; }
    public void setStudentConfig(Map<String, Object> value) { studentConfig = value; }
    public Map<String, Object> getProvenance() { return provenance; }
    public void setProvenance(Map<String, Object> value) { provenance = value; }
}
