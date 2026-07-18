/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * Content identity of the canonical SameDiff source model.
 *
 * <p>For SDZ, the digest covers sorted non-directory ZIP entries and their
 * uncompressed bytes, excluding {@code META-INF/sdx-cache/}. Consequently an
 * original SDZ and an SDZ enriched with target cache entries have the same
 * identity. Provider artifacts therefore never invalidate or recursively key
 * the canonical model.</p>
 */
public final class SdxSourceIdentity {
    static final String EMBEDDED_CACHE_ROOT = "META-INF/sdx-cache/";
    private static final byte[] SDZ_DOMAIN =
            "sdx-logical-sdz-v1\0".getBytes(StandardCharsets.UTF_8);
    private static final byte[] SDNB_DOMAIN =
            "sdx-logical-sdnb-v1\0".getBytes(StandardCharsets.UTF_8);

    private final String sha256;
    private final long logicalBytes;
    private final String sourceFileName;

    private SdxSourceIdentity(String sha256, long logicalBytes, String sourceFileName) {
        this.sha256 = sha256;
        this.logicalBytes = logicalBytes;
        this.sourceFileName = sourceFileName;
    }

    public static SdxSourceIdentity identify(Path sourceModel) throws IOException {
        Objects.requireNonNull(sourceModel, "sourceModel");
        Path source = sourceModel.toAbsolutePath().normalize();
        if (!Files.isRegularFile(source) || Files.size(source) <= 0L) {
            throw new IOException("SDX source model is missing or empty: " + source);
        }

        String name = source.getFileName().toString();
        String lower = name.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".sdz")) {
            return identifySdz(source);
        }
        if (lower.endsWith(".sdnb")) {
            return identifySdnb(source);
        }
        throw new IOException("SDX source model must end in .sdz or .sdnb: " + source);
    }

    private static SdxSourceIdentity identifySdz(Path source) throws IOException {
        MessageDigest digest = sha256Digest();
        digest.update(SDZ_DOMAIN);
        long logicalBytes = 0L;

        try (ZipFile zip = new ZipFile(source.toFile())) {
            List<? extends ZipEntry> entries = sortedSourceEntries(zip);
            byte[] buffer = new byte[1024 * 1024];
            for (ZipEntry entry : entries) {
                byte[] name = entry.getName().getBytes(StandardCharsets.UTF_8);
                digest.update(intBytes(name.length));
                digest.update(name);

                long declaredSize = entry.getSize();
                if (declaredSize < 0L) {
                    throw new IOException("SDZ entry has no declared size: " + entry.getName());
                }
                digest.update(longBytes(declaredSize));

                long count = 0L;
                try (InputStream input = zip.getInputStream(entry)) {
                    while (true) {
                        int read = input.read(buffer);
                        if (read < 0) {
                            break;
                        }
                        digest.update(buffer, 0, read);
                        count += read;
                    }
                }
                if (count != declaredSize) {
                    throw new IOException(
                            "SDZ entry size changed while reading " + entry.getName()
                                    + ": declared=" + declaredSize + ", actual=" + count);
                }
                logicalBytes = Math.addExact(logicalBytes, count);
            }
        } catch (ArithmeticException overflow) {
            throw new IOException("SDZ logical size overflow: " + source, overflow);
        }

        return new SdxSourceIdentity(hex(digest.digest()), logicalBytes, "model.sdz");
    }

    private static SdxSourceIdentity identifySdnb(Path source) throws IOException {
        MessageDigest digest = sha256Digest();
        digest.update(SDNB_DOMAIN);
        long size = Files.size(source);
        digest.update(longBytes(size));
        byte[] buffer = new byte[1024 * 1024];
        try (InputStream input = Files.newInputStream(source)) {
            while (true) {
                int read = input.read(buffer);
                if (read < 0) {
                    break;
                }
                digest.update(buffer, 0, read);
            }
        }
        return new SdxSourceIdentity(hex(digest.digest()), size, "model.sdnb");
    }

    static List<? extends ZipEntry> sortedSourceEntries(ZipFile zip) throws IOException {
        Enumeration<? extends ZipEntry> enumeration = zip.entries();
        List<ZipEntry> entries = new ArrayList<>();
        Set<String> names = new HashSet<>();
        while (enumeration.hasMoreElements()) {
            ZipEntry entry = enumeration.nextElement();
            String name = entry.getName();
            requireSafeEntryName(name);
            if (!names.add(name)) {
                throw new IOException("SDZ contains a duplicate ZIP entry: " + name);
            }
            if (!entry.isDirectory() && !name.startsWith(EMBEDDED_CACHE_ROOT)) {
                entries.add(entry);
            }
        }
        entries.sort(Comparator.comparing(ZipEntry::getName));
        return entries;
    }

    static void requireSafeEntryName(String name) throws IOException {
        if (name == null || name.isEmpty() || name.startsWith("/") || name.indexOf('\\') >= 0) {
            throw new IOException("Unsafe SDZ ZIP entry: " + name);
        }
        String normalized = name.endsWith("/") ? name.substring(0, name.length() - 1) : name;
        if (normalized.isEmpty()) {
            throw new IOException("Unsafe SDZ ZIP entry: " + name);
        }
        for (String part : normalized.split("/")) {
            if (part.isEmpty() || ".".equals(part) || "..".equals(part)) {
                throw new IOException("Unsafe SDZ ZIP entry: " + name);
            }
        }
    }

    public String sha256() {
        return sha256;
    }

    public long logicalBytes() {
        return logicalBytes;
    }

    public String sourceFileName() {
        return sourceFileName;
    }

    @Override
    public String toString() {
        return sha256;
    }

    private static MessageDigest sha256Digest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is required by the JDK", impossible);
        }
    }

    static String sha256(Path path) throws IOException {
        MessageDigest digest = sha256Digest();
        byte[] buffer = new byte[1024 * 1024];
        try (InputStream input = Files.newInputStream(path)) {
            while (true) {
                int read = input.read(buffer);
                if (read < 0) {
                    break;
                }
                digest.update(buffer, 0, read);
            }
        }
        return hex(digest.digest());
    }

    static byte[] sha256Bytes(byte[] value) {
        return sha256Digest().digest(value);
    }

    static String hex(byte[] value) {
        StringBuilder result = new StringBuilder(value.length * 2);
        for (byte b : value) {
            result.append(Character.forDigit((b >>> 4) & 0x0f, 16));
            result.append(Character.forDigit(b & 0x0f, 16));
        }
        return result.toString();
    }

    private static byte[] intBytes(int value) {
        return ByteBuffer.allocate(Integer.BYTES).putInt(value).array();
    }

    private static byte[] longBytes(long value) {
        return ByteBuffer.allocate(Long.BYTES).putLong(value).array();
    }
}
