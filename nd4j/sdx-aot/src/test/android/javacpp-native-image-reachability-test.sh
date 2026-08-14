#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR="$SCRIPT_DIR/../../main/android/JavaCppNativeImageReachability.java"
SCANNER="$SCRIPT_DIR/../../../../nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/nativeimage/Nd4jJavaCppClassScanner.java"

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -s "$GENERATOR" ]] || fail "missing reachability generator"
[[ -s "$SCANNER" ]] || fail "missing shared JavaCPP scanner"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf -- "$WORK_DIR"' EXIT
SOURCE_DIR="$WORK_DIR/src"
CLASSES_DIR="$WORK_DIR/classes"
OUTPUT_DIR="$WORK_DIR/output"
mkdir -p   "$SOURCE_DIR/org/bytedeco/javacpp"   "$SOURCE_DIR/org/nd4j/fixture"   "$SOURCE_DIR/org/nd4j/nativeimage"   "$CLASSES_DIR"   "$OUTPUT_DIR"

cp "$SCANNER" "$SOURCE_DIR/org/nd4j/nativeimage/Nd4jJavaCppClassScanner.java"

cat >"$SOURCE_DIR/org/bytedeco/javacpp/Pointer.java" <<'JAVA'
package org.bytedeco.javacpp;
public class Pointer {
}
JAVA

cat >"$SOURCE_DIR/org/bytedeco/javacpp/ClassProperties.java" <<'JAVA'
package org.bytedeco.javacpp;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
public class ClassProperties {
    private final Map<String, List<String>> values = new HashMap<>();
    private Class<?>[] inheritedClasses = new Class<?>[0];
    public List<String> get(String key) {
        return values.computeIfAbsent(key, ignored -> new ArrayList<>());
    }
    public Class<?>[] getInheritedClasses() {
        return inheritedClasses;
    }
    public void setInheritedClasses(Class<?>[] value) {
        inheritedClasses = value;
    }
}
JAVA

cat >"$SOURCE_DIR/org/bytedeco/javacpp/Loader.java" <<'JAVA'
package org.bytedeco.javacpp;
import java.util.Properties;
public final class Loader {
    private Loader() {
    }
    public static Properties loadProperties() {
        Properties properties = new Properties();
        properties.setProperty("platform", System.getProperty("org.bytedeco.javacpp.platform"));
        return properties;
    }
    public static ClassProperties loadProperties(
            Class<?> root, Properties ignored, boolean inherit) {
        ClassProperties properties = new ClassProperties();
        if (root.getName().equals("org.nd4j.fixture.BindingRoot")) {
            properties.get("global").add("org.nd4j.fixture.GlobalBinding");
        } else {
            properties.get("global").add(root.getName());
        }
        return properties;
    }
}
JAVA

cat >"$SOURCE_DIR/org/nd4j/fixture/BindingRoot.java" <<'JAVA'
package org.nd4j.fixture;
import org.bytedeco.javacpp.Pointer;
public class BindingRoot {
    public static native int nativeEntry();
    public static class Environment extends Pointer {
        public native int threshold();
    }
    public static class Vector extends Pointer {
        public static class Iterator extends Pointer {
            public native long position();
        }
    }
}
JAVA

cat >"$SOURCE_DIR/org/nd4j/fixture/GlobalBinding.java" <<'JAVA'
package org.nd4j.fixture;
import org.bytedeco.javacpp.Pointer;
public class GlobalBinding extends Pointer {
    public static native int globalEntry();
}
JAVA

javac -d "$CLASSES_DIR"   "$SOURCE_DIR/org/bytedeco/javacpp/Pointer.java"   "$SOURCE_DIR/org/bytedeco/javacpp/ClassProperties.java"   "$SOURCE_DIR/org/bytedeco/javacpp/Loader.java"   "$SOURCE_DIR/org/nd4j/nativeimage/Nd4jJavaCppClassScanner.java"   "$SOURCE_DIR/org/nd4j/fixture/BindingRoot.java"   "$SOURCE_DIR/org/nd4j/fixture/GlobalBinding.java"

printf '%s\n' "$CLASSES_DIR" >"$WORK_DIR/classpath.txt"

java -Dorg.bytedeco.javacpp.platform=android-arm64   "$GENERATOR"   "$WORK_DIR/classpath.txt"   "$OUTPUT_DIR/reflect-config.json"   "$OUTPUT_DIR/jni-config.json"   "$OUTPUT_DIR/native-image.properties"   "$OUTPUT_DIR/reachability.txt"

cmp -s "$OUTPUT_DIR/reflect-config.json" "$OUTPUT_DIR/jni-config.json" ||
  fail "reflection and JNI class catalogs diverged"

for class_name in   'org.nd4j.fixture.BindingRoot'   'org.nd4j.fixture.BindingRoot$Environment'   'org.nd4j.fixture.BindingRoot$Vector'   'org.nd4j.fixture.BindingRoot$Vector$Iterator'   'org.nd4j.fixture.GlobalBinding'; do
  grep -Fq "\"name\": \"$class_name\"" "$OUTPUT_DIR/reflect-config.json" ||
    fail "reflection metadata omits $class_name"
  grep -Fqx "reflection-class=$class_name" "$OUTPUT_DIR/reachability.txt" ||
    fail "reflection manifest omits $class_name"
  grep -Fqx "jni-class=$class_name" "$OUTPUT_DIR/reachability.txt" ||
    fail "JNI manifest omits $class_name"
  grep -Fq "$class_name" "$OUTPUT_DIR/native-image.properties" ||
    fail "runtime initialization metadata omits $class_name"
done

grep -Fqx 'format=2' "$OUTPUT_DIR/reachability.txt" ||
  fail "unexpected reachability manifest format"
grep -Fq '"allDeclaredMethods": true' "$OUTPUT_DIR/reflect-config.json" ||
  fail "method registration is incomplete"
grep -Fq '"allDeclaredConstructors": true' "$OUTPUT_DIR/reflect-config.json" ||
  fail "constructor registration is incomplete"
grep -Fq '"allDeclaredFields": true' "$OUTPUT_DIR/reflect-config.json" ||
  fail "field registration is incomplete"

printf 'PASS: complete recursive JavaCPP closure emitted for reflection, JNI, and runtime initialization\n'
