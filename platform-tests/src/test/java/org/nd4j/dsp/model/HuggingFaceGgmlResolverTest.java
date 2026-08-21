package org.nd4j.dsp.model;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HuggingFaceGgmlResolverTest {

    private static final String SHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    private static final String CONTENT_SHA256 =
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    @Test
    void parsesRepositoryAndCanonicalUrls() {
        HuggingFaceGgmlResolver.Reference repository =
                HuggingFaceGgmlResolver.parse(" owner/model ");
        assertEquals("owner/model", repository.getRepository());
        assertEquals("main", repository.getRequestedRevision());
        assertEquals(HuggingFaceGgmlResolver.Kind.REPOSITORY, repository.getKind());

        HuggingFaceGgmlResolver.Reference tree = HuggingFaceGgmlResolver.parse(
                "https://huggingface.co/owner/model/tree/release/quantized");
        assertEquals(HuggingFaceGgmlResolver.Kind.TREE, tree.getKind());
        assertEquals("release", tree.getRequestedRevision());
        assertEquals("quantized", tree.getRequestedPath());

        HuggingFaceGgmlResolver.Reference blob = HuggingFaceGgmlResolver.parse(
                "https://www.huggingface.co/owner/model/blob/v1/model-Q4_K_M.gguf");
        assertTrue(blob.isExactModel());
    }

    @Test
    void exactFileNeedsNoRepositoryListing() {
        HuggingFaceGgmlResolver.Discovery discovery = HuggingFaceGgmlResolver.exact(
                HuggingFaceGgmlResolver.parse(
                        "https://huggingface.co/owner/model/blob/main/weights/model.ggml"));

        assertFalse(discovery.requiresSelection());
        assertFalse(discovery.selectedCandidate().orElseThrow().isCommitPinned());
        assertEquals(
                "https://huggingface.co/owner/model/resolve/main/weights/model.ggml?download=true",
                discovery.selectedCandidate().orElseThrow().getDownloadUri().toASCIIString());
    }

    @Test
    void generatedResolveDownloadUrlRoundTripsWithoutBroadQuerySupport() {
        String generated = HuggingFaceGgmlResolver.exact(
                        HuggingFaceGgmlResolver.parse(
                                "https://huggingface.co/owner/model/blob/main/model.gguf"))
                .selectedCandidate()
                .orElseThrow()
                .getDownloadUri()
                .toASCIIString();

        HuggingFaceGgmlResolver.Reference roundTrip = HuggingFaceGgmlResolver.parse(generated);
        assertEquals(HuggingFaceGgmlResolver.Kind.RESOLVE, roundTrip.getKind());
        assertEquals(generated, HuggingFaceGgmlResolver.exact(roundTrip)
                .selectedCandidate().orElseThrow().getDownloadUri().toASCIIString());

        List<String> unsafeQueries = List.of(
                "https://huggingface.co/owner/model?download=true",
                "https://huggingface.co/owner/model/blob/main/model.gguf?download=true",
                "https://huggingface.co/owner/model/resolve/main/model.gguf?token=secret",
                "https://huggingface.co/owner/model/resolve/main/model.gguf?download=true&download=true");
        unsafeQueries.forEach(value -> assertThrows(
                IllegalArgumentException.class,
                () -> HuggingFaceGgmlResolver.parse(value)));
    }

    @Test
    void repositoryNameDiscoversAndPinsOneGguf() {
        HuggingFaceGgmlResolver.Reference reference = HuggingFaceGgmlResolver.parse("owner/model");
        HuggingFaceGgmlResolver.Discovery discovery = HuggingFaceGgmlResolver.resolve(
                reference,
                SHA,
                List.of(
                        new HuggingFaceGgmlResolver.RepositoryFile("config.json", 20),
                        new HuggingFaceGgmlResolver.RepositoryFile(
                                "model-Q4_K_M.gguf", 1234, CONTENT_SHA256)));

        assertFalse(discovery.requiresSelection());
        HuggingFaceGgmlResolver.Candidate selected = discovery.selectedCandidate().orElseThrow();
        assertTrue(selected.isCommitPinned());
        assertEquals(CONTENT_SHA256, selected.getSha256());
        assertEquals("Q4_K_M", selected.getQuantizationHint());
        assertEquals(
                "https://huggingface.co/owner/model/resolve/" + SHA
                        + "/model-Q4_K_M.gguf?download=true",
                selected.getDownloadUri().toASCIIString());
    }

    @Test
    void candidateCarriesTokenizerAssetsFromTheSameImmutableCommit() {
        String tokenizerSha =
                "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        HuggingFaceGgmlResolver.Candidate selected = HuggingFaceGgmlResolver.resolve(
                HuggingFaceGgmlResolver.parse("owner/model"),
                SHA,
                List.of(
                        new HuggingFaceGgmlResolver.RepositoryFile("weights/model.gguf", 1234),
                        new HuggingFaceGgmlResolver.RepositoryFile("tokenizer.json", 50, tokenizerSha),
                        new HuggingFaceGgmlResolver.RepositoryFile("tokenizer_config.json", 60),
                        new HuggingFaceGgmlResolver.RepositoryFile("config.json", 40),
                        new HuggingFaceGgmlResolver.RepositoryFile("chat_template.jinja", 30),
                        new HuggingFaceGgmlResolver.RepositoryFile("README.md", 70)))
                .selectedCandidate().orElseThrow();

        assertEquals(4, selected.getTokenizerAssets().size());
        HuggingFaceGgmlResolver.TokenizerAsset tokenizer = selected.getTokenizerAssets().get(0);
        assertEquals("tokenizer.json", tokenizer.getName());
        assertEquals("tokenizer.json", tokenizer.getPath());
        assertEquals(tokenizerSha, tokenizer.getSha256());
        assertEquals(
                "https://huggingface.co/owner/model/resolve/" + SHA
                        + "/tokenizer.json?download=true",
                tokenizer.getDownloadUri().toASCIIString());
        assertEquals("chat_template.jinja", selected.getTokenizerAssets().get(2).getName());
        assertEquals("config.json", selected.getTokenizerAssets().get(3).getName());
    }

    @Test
    void candidateUsesSeparatelyPinnedBaseModelConfigurationWithoutMovingWeights() {
        String configurationSha = "dddddddddddddddddddddddddddddddddddddddd";
        HuggingFaceGgmlResolver.Discovery discovery = HuggingFaceGgmlResolver.resolve(
                HuggingFaceGgmlResolver.parse("unsloth/Qwen3.5-0.8B-GGUF"),
                SHA,
                List.of(new HuggingFaceGgmlResolver.RepositoryFile(
                        "Qwen3.5-0.8B-Q4_K_M.gguf", 1234, CONTENT_SHA256)),
                "Qwen/Qwen3.5-0.8B",
                configurationSha,
                List.of(
                        new HuggingFaceGgmlResolver.RepositoryFile("tokenizer.json", 50),
                        new HuggingFaceGgmlResolver.RepositoryFile("tokenizer_config.json", 60),
                        new HuggingFaceGgmlResolver.RepositoryFile("config.json", 40),
                        new HuggingFaceGgmlResolver.RepositoryFile("chat_template.jinja", 30)));

        HuggingFaceGgmlResolver.Candidate selected =
                discovery.selectedCandidate().orElseThrow();
        assertEquals("unsloth/Qwen3.5-0.8B-GGUF", discovery.getReference().getRepository());
        assertEquals(SHA, discovery.getResolvedRevision());
        assertEquals(1, discovery.getAssetSources().size());
        assertEquals("Qwen/Qwen3.5-0.8B", discovery.getAssetSources().get(0).getRepository());
        assertEquals(configurationSha, discovery.getAssetSources().get(0).getResolvedRevision());
        assertEquals(
                "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/" + SHA
                        + "/Qwen3.5-0.8B-Q4_K_M.gguf?download=true",
                selected.getDownloadUri().toASCIIString());
        assertEquals(
                "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/" + configurationSha
                        + "/tokenizer.json?download=true",
                selected.getTokenizerAssets().get(0).getDownloadUri().toASCIIString());
        assertEquals(
                "Qwen/Qwen3.5-0.8B",
                selected.getTokenizerAssets().get(0).getSourceRepository());
        assertEquals(
                configurationSha,
                selected.getTokenizerAssets().get(0).getSourceRevision());
    }

    @Test
    void resolvesEachCanonicalAssetFromItsNearestPinnedUpstreamRepository() {
        String configurationSha = "dddddddddddddddddddddddddddddddddddddddd";
        String tokenizerSha = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
        HuggingFaceGgmlResolver.Discovery discovery = HuggingFaceGgmlResolver.resolve(
                HuggingFaceGgmlResolver.parse("vendor/chat-GGUF"),
                SHA,
                List.of(new HuggingFaceGgmlResolver.RepositoryFile("chat-Q4.gguf", 1234)),
                List.of(
                        new HuggingFaceGgmlResolver.RepositorySnapshot(
                                "vendor/chat-GGUF",
                                SHA,
                                List.of(
                                        new HuggingFaceGgmlResolver.RepositoryFile(
                                                "chat-Q4.gguf", 1234),
                                        new HuggingFaceGgmlResolver.RepositoryFile(
                                                "generation_config.json", 12))),
                        new HuggingFaceGgmlResolver.RepositorySnapshot(
                                "vendor/chat-config",
                                configurationSha,
                                List.of(
                                        new HuggingFaceGgmlResolver.RepositoryFile("config.json", 42),
                                        new HuggingFaceGgmlResolver.RepositoryFile(
                                                "chat_template.jinja", 20))),
                        new HuggingFaceGgmlResolver.RepositorySnapshot(
                                "vendor/chat-tokenizer",
                                tokenizerSha,
                                List.of(
                                        new HuggingFaceGgmlResolver.RepositoryFile(
                                                "tokenizer.json", 100),
                                        new HuggingFaceGgmlResolver.RepositoryFile(
                                                "tokenizer_config.json", 30),
                                        new HuggingFaceGgmlResolver.RepositoryFile(
                                                "added_tokens.json", 10)))));

        HuggingFaceGgmlResolver.Candidate candidate =
                discovery.selectedCandidate().orElseThrow();
        assertEquals(
                List.of("vendor/chat-GGUF", "vendor/chat-config", "vendor/chat-tokenizer"),
                discovery.getAssetSources().stream()
                        .map(HuggingFaceGgmlResolver.AssetSource::getRepository)
                        .collect(Collectors.toList()));
        HuggingFaceGgmlResolver.TokenizerAsset tokenizer = candidate.getTokenizerAssets().stream()
                .filter(asset -> "tokenizer.json".equals(asset.getName()))
                .findFirst().orElseThrow();
        HuggingFaceGgmlResolver.TokenizerAsset config = candidate.getTokenizerAssets().stream()
                .filter(asset -> "config.json".equals(asset.getName()))
                .findFirst().orElseThrow();
        HuggingFaceGgmlResolver.TokenizerAsset generation = candidate.getTokenizerAssets().stream()
                .filter(asset -> "generation_config.json".equals(asset.getName()))
                .findFirst().orElseThrow();
        assertEquals("vendor/chat-tokenizer", tokenizer.getSourceRepository());
        assertEquals(tokenizerSha, tokenizer.getSourceRevision());
        assertEquals("vendor/chat-config", config.getSourceRepository());
        assertEquals(configurationSha, config.getSourceRevision());
        assertEquals("vendor/chat-GGUF", generation.getSourceRepository());
        assertEquals(SHA, generation.getSourceRevision());
    }

    @Test
    void repositoryRootConfigurationIsSharedAcrossEveryModelCandidate() {
        HuggingFaceGgmlResolver.Discovery discovery = HuggingFaceGgmlResolver.resolve(
                HuggingFaceGgmlResolver.parse("owner/model"),
                SHA,
                List.of(
                        new HuggingFaceGgmlResolver.RepositoryFile("mobile/model-Q4.gguf", 1234),
                        new HuggingFaceGgmlResolver.RepositoryFile("desktop/model-Q8.gguf", 2234),
                        new HuggingFaceGgmlResolver.RepositoryFile("tokenizer.json", 10),
                        new HuggingFaceGgmlResolver.RepositoryFile("tokenizer_config.json", 30),
                        new HuggingFaceGgmlResolver.RepositoryFile("config.json", 40),
                        new HuggingFaceGgmlResolver.RepositoryFile("mobile/tokenizer.json", 20),
                        new HuggingFaceGgmlResolver.RepositoryFile("added_tokens.json", 15),
                        new HuggingFaceGgmlResolver.RepositoryFile("text-generation.json", 25)));

        assertEquals(2, discovery.getCandidates().size());
        List<String> expectedPaths = List.of(
                "tokenizer.json", "tokenizer_config.json", "added_tokens.json",
                "config.json", "text-generation.json");
        for (HuggingFaceGgmlResolver.Candidate candidate : discovery.getCandidates()) {
            assertEquals(expectedPaths, candidate.getTokenizerAssets().stream()
                    .map(HuggingFaceGgmlResolver.TokenizerAsset::getPath)
                    .collect(Collectors.toList()));
        }
    }

    @Test
    void ambiguousNestedConfigurationFailsInsteadOfChoosingAFlavorDirectory() {
        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> HuggingFaceGgmlResolver.resolve(
                        HuggingFaceGgmlResolver.parse("owner/model"),
                        SHA,
                        List.of(
                                new HuggingFaceGgmlResolver.RepositoryFile("mobile/model.gguf", 1234),
                                new HuggingFaceGgmlResolver.RepositoryFile("mobile/tokenizer.json", 20),
                                new HuggingFaceGgmlResolver.RepositoryFile("desktop/tokenizer.json", 21))));

        assertTrue(failure.getMessage().contains("ambiguous tokenizer.json"));
        assertTrue(failure.getMessage().contains("desktop/tokenizer.json"));
        assertTrue(failure.getMessage().contains("mobile/tokenizer.json"));
    }

    @Test
    void supportsEncodedFilenamesAndEncodedSlashBearingRevisions() {
        HuggingFaceGgmlResolver.Reference reference = HuggingFaceGgmlResolver.parse(
                "https://huggingface.co/owner/model/tree/refs%2Fpr%2F123/weights%20set");

        assertEquals("refs/pr/123", reference.getRequestedRevision());
        assertEquals("weights set", reference.getRequestedPath());
        assertEquals(
                "https://huggingface.co/api/models/owner/model/revision/refs%2Fpr%2F123?blobs=true",
                HuggingFaceGgmlResolver.apiUri(reference).toASCIIString());

        HuggingFaceGgmlResolver.Discovery discovery = HuggingFaceGgmlResolver.resolve(
                reference,
                SHA,
                List.of(new HuggingFaceGgmlResolver.RepositoryFile(
                        "weights set/model Q4_K_M.gguf", 42)));
        assertEquals(
                "https://huggingface.co/owner/model/resolve/" + SHA
                        + "/weights%20set/model%20Q4_K_M.gguf?download=true",
                discovery.selectedCandidate().orElseThrow().getDownloadUri().toASCIIString());

        HuggingFaceGgmlResolver.Discovery exact = HuggingFaceGgmlResolver.exact(
                HuggingFaceGgmlResolver.parse(
                        "https://huggingface.co/owner/model/resolve/refs%2Fpr%2F123/"
                                + "model%20Q4.gguf?download=true"));
        assertEquals("refs/pr/123", exact.getResolvedRevision());
        assertEquals("model Q4.gguf", exact.selectedCandidate().orElseThrow().getPath());
        assertEquals(
                "https://huggingface.co/owner/model/resolve/refs%2Fpr%2F123/"
                        + "model%20Q4.gguf?download=true",
                exact.selectedCandidate().orElseThrow().getDownloadUri().toASCIIString());
    }

    @Test
    void splitGgufShardsAreNeverOfferedAsStandaloneDownloads() {
        assertThrows(
                IllegalArgumentException.class,
                () -> HuggingFaceGgmlResolver.parse(
                        "https://huggingface.co/owner/model/resolve/main/"
                                + "model-00001-of-00003.gguf"));

        IllegalArgumentException onlyShards = assertThrows(
                IllegalArgumentException.class,
                () -> HuggingFaceGgmlResolver.resolve(
                        HuggingFaceGgmlResolver.parse("owner/model"),
                        SHA,
                        List.of(
                                new HuggingFaceGgmlResolver.RepositoryFile(
                                        "model-00001-of-00002.gguf", 1),
                                new HuggingFaceGgmlResolver.RepositoryFile(
                                        "model-00002-of-00002.gguf", 1))));
        assertTrue(onlyShards.getMessage().contains("Split GGUF shard"));

        HuggingFaceGgmlResolver.Discovery mixed = HuggingFaceGgmlResolver.resolve(
                HuggingFaceGgmlResolver.parse("owner/model"),
                SHA,
                List.of(
                        new HuggingFaceGgmlResolver.RepositoryFile(
                                "model-00001-of-00002.gguf", 1),
                        new HuggingFaceGgmlResolver.RepositoryFile("model-Q4_K_M.gguf", 2)));
        assertEquals(1, mixed.getCandidates().size());
        assertEquals("model-Q4_K_M.gguf", mixed.getCandidates().get(0).getPath());
    }

    @Test
    void multipleQuantizationsRemainExplicitAndSorted() {
        HuggingFaceGgmlResolver.Discovery discovery = HuggingFaceGgmlResolver.resolve(
                HuggingFaceGgmlResolver.parse("owner/model"),
                SHA,
                List.of(
                        new HuggingFaceGgmlResolver.RepositoryFile("z/model-Q8_0.gguf", 300),
                        new HuggingFaceGgmlResolver.RepositoryFile("a/model-Q4_K_M.gguf", 200)));

        assertTrue(discovery.requiresSelection());
        assertTrue(discovery.selectedCandidate().isEmpty());
        assertEquals("a/model-Q4_K_M.gguf", discovery.getCandidates().get(0).getPath());
        assertEquals("z/model-Q8_0.gguf", discovery.getCandidates().get(1).getPath());
    }

    @Test
    void treeReferenceScopesCandidates() {
        HuggingFaceGgmlResolver.Discovery discovery = HuggingFaceGgmlResolver.resolve(
                HuggingFaceGgmlResolver.parse(
                        "https://huggingface.co/owner/model/tree/main/mobile"),
                SHA,
                List.of(
                        new HuggingFaceGgmlResolver.RepositoryFile("desktop/model.gguf", 300),
                        new HuggingFaceGgmlResolver.RepositoryFile("mobile/model.ggml", 200)));

        assertEquals(1, discovery.getCandidates().size());
        assertEquals("mobile/model.ggml", discovery.getCandidates().get(0).getPath());
    }

    @Test
    void rejectsUnsafeReferencesAndMalformedListings() {
        List<String> unsafe = List.of(
                "owner/model/extra",
                "http://huggingface.co/owner/model",
                "https://token@huggingface.co/owner/model",
                "https://huggingface.co:443/owner/model",
                "https://huggingface.co/owner/model?token=secret",
                "https://huggingface.co/owner/model/blob/main/a%2Fb.gguf",
                "https://huggingface.co/owner/model/blob/main/%2E%2E/model.gguf",
                "https://huggingface.co/owner/model/repository/main/model.gguf",
                "https://example.com/owner/model");
        unsafe.forEach(value -> assertThrows(
                IllegalArgumentException.class,
                () -> HuggingFaceGgmlResolver.parse(value)));

        assertThrows(
                IllegalArgumentException.class,
                () -> HuggingFaceGgmlResolver.resolve(
                        HuggingFaceGgmlResolver.parse("owner/model"),
                        "main",
                        List.of(new HuggingFaceGgmlResolver.RepositoryFile("model.gguf", 1))));
        assertThrows(
                IllegalArgumentException.class,
                () -> new HuggingFaceGgmlResolver.RepositoryFile("../model.gguf", 1));
        assertThrows(
                IllegalArgumentException.class,
                () -> new HuggingFaceGgmlResolver.RepositoryFile(
                        "model.gguf", 1, "not-a-sha256"));
    }

    @Test
    void rejectsEmptyAndDuplicateCandidateSets() {
        HuggingFaceGgmlResolver.Reference reference = HuggingFaceGgmlResolver.parse("owner/model");
        assertThrows(
                IllegalArgumentException.class,
                () -> HuggingFaceGgmlResolver.resolve(
                        reference,
                        SHA,
                        List.of(new HuggingFaceGgmlResolver.RepositoryFile("config.json", 1))));
        assertThrows(
                IllegalArgumentException.class,
                () -> HuggingFaceGgmlResolver.resolve(
                        reference,
                        SHA,
                        List.of(
                                new HuggingFaceGgmlResolver.RepositoryFile("model.gguf", 1),
                                new HuggingFaceGgmlResolver.RepositoryFile("model.gguf", 1))));
    }
}
