/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <dsp/runtime/detail/SdxTextGenerationMetadata.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sd {
namespace dsp {
namespace runtime {
namespace detail {
namespace {

enum class JsonKind { Null, Boolean, Number, String, Array, Object };

struct JsonValue {
  JsonKind kind = JsonKind::Null;
  bool boolean = false;
  double number = 0.0;
  std::string string;
  std::vector<JsonValue> array;
  std::map<std::string, JsonValue> object;
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& input) : _input(input) {}

  bool parse(JsonValue* out, std::string* error) {
    if (out == nullptr) return fail("output is null", error);
    skipWhitespace();
    if (!parseValue(out, error)) return false;
    skipWhitespace();
    if (_position != _input.size()) {
      return fail("unexpected trailing content", error);
    }
    return true;
  }

 private:
  bool parseValue(JsonValue* out, std::string* error) {
    skipWhitespace();
    if (_position >= _input.size()) return fail("unexpected end of input", error);

    const char c = _input[_position];
    if (c == '{') return parseObject(out, error);
    if (c == '[') return parseArray(out, error);
    if (c == '"') {
      out->kind = JsonKind::String;
      return parseString(&out->string, error);
    }
    if (c == 't') return parseLiteral("true", JsonKind::Boolean, true, out, error);
    if (c == 'f') return parseLiteral("false", JsonKind::Boolean, false, out, error);
    if (c == 'n') return parseLiteral("null", JsonKind::Null, false, out, error);
    if (c == '-' || (c >= '0' && c <= '9')) return parseNumber(out, error);
    return fail(std::string("unexpected character '") + c + "'", error);
  }

  bool parseObject(JsonValue* out, std::string* error) {
    ++_position;
    out->kind = JsonKind::Object;
    out->object.clear();
    skipWhitespace();
    if (consume('}')) return true;

    while (true) {
      skipWhitespace();
      std::string key;
      if (!parseString(&key, error)) return false;
      skipWhitespace();
      if (!consume(':')) return fail("expected ':' after object key", error);
      JsonValue value;
      if (!parseValue(&value, error)) return false;
      if (!out->object.emplace(key, std::move(value)).second) {
        return fail("duplicate object key '" + key + "'", error);
      }
      skipWhitespace();
      if (consume('}')) return true;
      if (!consume(',')) return fail("expected ',' or '}' in object", error);
    }
  }

  bool parseArray(JsonValue* out, std::string* error) {
    ++_position;
    out->kind = JsonKind::Array;
    out->array.clear();
    skipWhitespace();
    if (consume(']')) return true;

    while (true) {
      JsonValue value;
      if (!parseValue(&value, error)) return false;
      out->array.emplace_back(std::move(value));
      skipWhitespace();
      if (consume(']')) return true;
      if (!consume(',')) return fail("expected ',' or ']' in array", error);
    }
  }

  static void appendUtf8(uint32_t codePoint, std::string* out) {
    if (codePoint <= 0x7f) {
      out->push_back(static_cast<char>(codePoint));
    } else if (codePoint <= 0x7ff) {
      out->push_back(static_cast<char>(0xc0 | (codePoint >> 6)));
      out->push_back(static_cast<char>(0x80 | (codePoint & 0x3f)));
    } else if (codePoint <= 0xffff) {
      out->push_back(static_cast<char>(0xe0 | (codePoint >> 12)));
      out->push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3f)));
      out->push_back(static_cast<char>(0x80 | (codePoint & 0x3f)));
    } else {
      out->push_back(static_cast<char>(0xf0 | (codePoint >> 18)));
      out->push_back(static_cast<char>(0x80 | ((codePoint >> 12) & 0x3f)));
      out->push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3f)));
      out->push_back(static_cast<char>(0x80 | (codePoint & 0x3f)));
    }
  }

  bool parseHex4(uint32_t* out, std::string* error) {
    if (_position + 4 > _input.size()) return fail("truncated unicode escape", error);
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = _input[_position++];
      value <<= 4;
      if (c >= '0' && c <= '9') {
        value |= static_cast<uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        value |= static_cast<uint32_t>(10 + c - 'a');
      } else if (c >= 'A' && c <= 'F') {
        value |= static_cast<uint32_t>(10 + c - 'A');
      } else {
        return fail("invalid unicode escape", error);
      }
    }
    *out = value;
    return true;
  }

  bool parseString(std::string* out, std::string* error) {
    if (!consume('"')) return fail("expected JSON string", error);
    out->clear();

    while (_position < _input.size()) {
      const unsigned char c = static_cast<unsigned char>(_input[_position++]);
      if (c == '"') return true;
      if (c < 0x20) return fail("unescaped control character in string", error);
      if (c != '\\') {
        out->push_back(static_cast<char>(c));
        continue;
      }

      if (_position >= _input.size()) return fail("truncated string escape", error);
      const char escaped = _input[_position++];
      switch (escaped) {
        case '"': out->push_back('"'); break;
        case '\\': out->push_back('\\'); break;
        case '/': out->push_back('/'); break;
        case 'b': out->push_back('\b'); break;
        case 'f': out->push_back('\f'); break;
        case 'n': out->push_back('\n'); break;
        case 'r': out->push_back('\r'); break;
        case 't': out->push_back('\t'); break;
        case 'u': {
          uint32_t codePoint = 0;
          if (!parseHex4(&codePoint, error)) return false;
          if (codePoint >= 0xd800 && codePoint <= 0xdbff) {
            if (_position + 2 > _input.size() || _input[_position] != '\\' ||
                _input[_position + 1] != 'u') {
              return fail("high surrogate without low surrogate", error);
            }
            _position += 2;
            uint32_t low = 0;
            if (!parseHex4(&low, error)) return false;
            if (low < 0xdc00 || low > 0xdfff) {
              return fail("invalid low surrogate", error);
            }
            codePoint =
                0x10000 + ((codePoint - 0xd800) << 10) + (low - 0xdc00);
          } else if (codePoint >= 0xdc00 && codePoint <= 0xdfff) {
            return fail("unexpected low surrogate", error);
          }
          appendUtf8(codePoint, out);
          break;
        }
        default:
          return fail(std::string("unsupported string escape '\\") + escaped + "'", error);
      }
    }
    return fail("unterminated JSON string", error);
  }

  bool parseNumber(JsonValue* out, std::string* error) {
    const size_t start = _position;
    if (_input[_position] == '-') ++_position;
    if (_position >= _input.size()) return fail("truncated number", error);

    if (_input[_position] == '0') {
      ++_position;
    } else {
      if (_input[_position] < '1' || _input[_position] > '9') {
        return fail("invalid number", error);
      }
      while (_position < _input.size() &&
             _input[_position] >= '0' && _input[_position] <= '9') {
        ++_position;
      }
    }

    if (_position < _input.size() && _input[_position] == '.') {
      ++_position;
      const size_t fractionStart = _position;
      while (_position < _input.size() &&
             _input[_position] >= '0' && _input[_position] <= '9') {
        ++_position;
      }
      if (_position == fractionStart) return fail("empty number fraction", error);
    }

    if (_position < _input.size() &&
        (_input[_position] == 'e' || _input[_position] == 'E')) {
      ++_position;
      if (_position < _input.size() &&
          (_input[_position] == '+' || _input[_position] == '-')) {
        ++_position;
      }
      const size_t exponentStart = _position;
      while (_position < _input.size() &&
             _input[_position] >= '0' && _input[_position] <= '9') {
        ++_position;
      }
      if (_position == exponentStart) return fail("empty number exponent", error);
    }

    const std::string text = _input.substr(start, _position - start);
    char* end = nullptr;
    const double number = std::strtod(text.c_str(), &end);
    if (end == nullptr || *end != '\0' || !std::isfinite(number)) {
      return fail("invalid finite number", error);
    }
    out->kind = JsonKind::Number;
    out->number = number;
    return true;
  }

  bool parseLiteral(const char* literal, JsonKind kind, bool boolean,
                    JsonValue* out, std::string* error) {
    const size_t length = std::char_traits<char>::length(literal);
    if (_input.compare(_position, length, literal) != 0) {
      return fail(std::string("expected '") + literal + "'", error);
    }
    _position += length;
    out->kind = kind;
    out->boolean = boolean;
    return true;
  }

  bool consume(char expected) {
    if (_position < _input.size() && _input[_position] == expected) {
      ++_position;
      return true;
    }
    return false;
  }

  void skipWhitespace() {
    while (_position < _input.size()) {
      const char c = _input[_position];
      if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
      ++_position;
    }
  }

  bool fail(const std::string& message, std::string* error) const {
    if (error != nullptr) {
      *error = "JSON parse error at byte " + std::to_string(_position) +
               ": " + message;
    }
    return false;
  }

  const std::string& _input;
  size_t _position = 0;
};

const JsonValue* member(const JsonValue& object, const char* name) {
  if (object.kind != JsonKind::Object) return nullptr;
  const auto it = object.object.find(name);
  return it == object.object.end() ? nullptr : &it->second;
}

bool requireObject(const JsonValue& parent, const char* name,
                   const JsonValue** out, std::string* error) {
  const JsonValue* value = member(parent, name);
  if (value == nullptr || value->kind != JsonKind::Object) {
    if (error != nullptr) *error = std::string("metadata field '") + name + "' must be an object";
    return false;
  }
  *out = value;
  return true;
}

bool readString(const JsonValue& parent, const char* name, bool required,
                std::string* out, std::string* error) {
  const JsonValue* value = member(parent, name);
  if (value == nullptr) {
    if (!required) return true;
    if (error != nullptr) *error = std::string("metadata field '") + name + "' is required";
    return false;
  }
  if (value->kind != JsonKind::String || (required && value->string.empty())) {
    if (error != nullptr) *error = std::string("metadata field '") + name + "' must be a non-empty string";
    return false;
  }
  *out = value->string;
  return true;
}

bool readBoolean(const JsonValue& parent, const char* name, bool required,
                 bool* out, std::string* error) {
  const JsonValue* value = member(parent, name);
  if (value == nullptr) {
    if (!required) return true;
    if (error != nullptr) *error = std::string("metadata field '") + name + "' is required";
    return false;
  }
  if (value->kind != JsonKind::Boolean) {
    if (error != nullptr) *error = std::string("metadata field '") + name + "' must be boolean";
    return false;
  }
  *out = value->boolean;
  return true;
}

bool readNumber(const JsonValue& parent, const char* name, bool required,
                double* out, std::string* error) {
  const JsonValue* value = member(parent, name);
  if (value == nullptr) {
    if (!required) return true;
    if (error != nullptr) *error = std::string("metadata field '") + name + "' is required";
    return false;
  }
  if (value->kind != JsonKind::Number) {
    if (error != nullptr) *error = std::string("metadata field '") + name + "' must be numeric";
    return false;
  }
  *out = value->number;
  return true;
}

bool readInt64(const JsonValue& parent, const char* name, bool required,
               int64_t* out, std::string* error) {
  double value = static_cast<double>(*out);
  if (!readNumber(parent, name, required, &value, error)) return false;
  if (member(parent, name) == nullptr && !required) return true;
  if (std::floor(value) != value ||
      value < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
      value > static_cast<double>(std::numeric_limits<int64_t>::max())) {
    if (error != nullptr) *error = std::string("metadata field '") + name + "' must be an integer";
    return false;
  }
  *out = static_cast<int64_t>(value);
  return true;
}

bool readInt32(const JsonValue& parent, const char* name, bool required,
               int32_t* out, std::string* error) {
  int64_t value = *out;
  if (!readInt64(parent, name, required, &value, error)) return false;
  if (member(parent, name) == nullptr && !required) return true;
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max()) {
    if (error != nullptr) *error = std::string("metadata field '") + name + "' exceeds int32";
    return false;
  }
  *out = static_cast<int32_t>(value);
  return true;
}

bool readStringArray(const JsonValue& parent, const char* name,
                     std::vector<std::string>* out, std::string* error) {
  const JsonValue* value = member(parent, name);
  if (value == nullptr || value->kind != JsonKind::Array || value->array.empty()) {
    if (error != nullptr) *error = std::string("metadata field '") + name + "' must be a non-empty string array";
    return false;
  }
  out->clear();
  for (const auto& item : value->array) {
    if (item.kind != JsonKind::String || item.string.empty()) {
      if (error != nullptr) *error = std::string("metadata field '") + name + "' contains an invalid string";
      return false;
    }
    out->push_back(item.string);
  }
  return true;
}

bool readIntArray(const JsonValue& parent, const char* name,
                  std::vector<int>* out, std::string* error) {
  const JsonValue* value = member(parent, name);
  if (value == nullptr || value->kind != JsonKind::Array || value->array.empty()) {
    if (error != nullptr) *error = std::string("metadata field '") + name + "' must be a non-empty integer array";
    return false;
  }
  out->clear();
  for (const auto& item : value->array) {
    if (item.kind != JsonKind::Number || std::floor(item.number) != item.number ||
        item.number < 0 ||
        item.number > static_cast<double>(std::numeric_limits<int>::max())) {
      if (error != nullptr) *error = std::string("metadata field '") + name + "' contains an invalid token ID";
      return false;
    }
    out->push_back(static_cast<int>(item.number));
  }
  std::sort(out->begin(), out->end());
  if (std::adjacent_find(out->begin(), out->end()) != out->end()) {
    if (error != nullptr) *error = std::string("metadata field '") + name + "' contains duplicate token IDs";
    return false;
  }
  return true;
}

bool readKvShapeTemplates(
    const JsonValue& parent,
    const char* name,
    std::vector<std::vector<int64_t>>* out,
    std::string* error) {
  const JsonValue* value = member(parent, name);
  if (value == nullptr || value->kind != JsonKind::Array || value->array.empty()) {
    if (error != nullptr) {
      *error = std::string("metadata field '") + name +
               "' must be a non-empty array of BSHD shapes";
    }
    return false;
  }
  out->clear();
  for (size_t index = 0; index < value->array.size(); ++index) {
    const JsonValue& shape = value->array[index];
    if (shape.kind != JsonKind::Array || shape.array.size() != 4) {
      if (error != nullptr) {
        *error = std::string("metadata field '") + name + "[" +
                 std::to_string(index) + "]' must be a rank-4 BSHD shape";
      }
      return false;
    }
    std::vector<int64_t> dimensions;
    dimensions.reserve(4);
    for (const auto& dimension : shape.array) {
      if (dimension.kind != JsonKind::Number ||
          !std::isfinite(dimension.number) ||
          std::floor(dimension.number) != dimension.number ||
          dimension.number < -1.0 ||
          dimension.number >= 9223372036854775808.0) {
        if (error != nullptr) {
          *error = std::string("metadata field '") + name + "[" +
                   std::to_string(index) + "]' contains an invalid dimension";
        }
        return false;
      }
      dimensions.push_back(static_cast<int64_t>(dimension.number));
    }
    if (dimensions[0] != 1 || dimensions[1] != -1 ||
        dimensions[2] <= 0 || dimensions[3] <= 0) {
      if (error != nullptr) {
        *error = std::string("metadata field '") + name + "[" +
                 std::to_string(index) +
                 "]' must be [1,-1,numKvHeads,headDim]";
      }
      return false;
    }
    out->push_back(std::move(dimensions));
  }
  return true;
}

bool readOptionalKvShapeTemplates(
    const JsonValue& parent,
    const char* name,
    std::vector<std::vector<int64_t>>* out,
    std::string* error) {
  const JsonValue* value = member(parent, name);
  if (value == nullptr ||
      (value->kind == JsonKind::Array && value->array.empty())) {
    out->clear();
    return true;
  }
  return readKvShapeTemplates(parent, name, out, error);
}

std::string upper(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return value;
}

bool parseDataType(const std::string& raw, DataType* out) {
  const std::string value = upper(raw);
  if (value == "FLOAT32" || value == "FLOAT" || value == "FP32") {
    *out = DataType::FLOAT32;
    return true;
  }
  if (value == "FLOAT16" || value == "HALF" || value == "FP16") {
    *out = DataType::HALF;
    return true;
  }
  if (value == "BFLOAT16" || value == "BF16") {
    *out = DataType::BFLOAT16;
    return true;
  }
  if (value == "INT8") {
    *out = DataType::INT8;
    return true;
  }
  return false;
}

bool parseCanonicalRecurrentDataType(
    const std::string& raw, DataType* out) {
  if (raw == "FLOAT32") {
    *out = DataType::FLOAT32;
    return true;
  }
  if (raw == "FLOAT16") {
    *out = DataType::HALF;
    return true;
  }
  if (raw == "BFLOAT16") {
    *out = DataType::BFLOAT16;
    return true;
  }
  if (raw == "INT8") {
    *out = DataType::INT8;
    return true;
  }
  return false;
}

bool parseRecurrentStates(
    const JsonValue& io,
    int32_t formatVersion,
    std::vector<TextGenerationRecurrentState>* out,
    std::string* error) {
  const JsonValue* states = member(io, "recurrentStates");
  if (formatVersion == 1) {
    if (states != nullptr) {
      if (error != nullptr) {
        *error =
            "metadata field 'io.recurrentStates' requires text-generation "
            "formatVersion 2";
      }
      return false;
    }
    out->clear();
    return true;
  }
  if (states == nullptr) {
    if (error != nullptr) {
      *error =
          "metadata field 'io.recurrentStates' is required for formatVersion 2";
    }
    return false;
  }
  if (states->kind != JsonKind::Array) {
    if (error != nullptr) {
      *error = "metadata field 'io.recurrentStates' must be an array";
    }
    return false;
  }

  out->clear();
  std::unordered_set<std::string> inputs;
  std::unordered_set<std::string> outputs;
  static const std::unordered_set<std::string> supportedFields{
      "input", "output", "kind", "dataType", "shape"};

  for (size_t index = 0; index < states->array.size(); ++index) {
    const JsonValue& value = states->array[index];
    const std::string prefix =
        "metadata field 'io.recurrentStates[" + std::to_string(index) + "]'";
    if (value.kind != JsonKind::Object) {
      if (error != nullptr) *error = prefix + " must be an object";
      return false;
    }
    for (const auto& entry : value.object) {
      if (supportedFields.count(entry.first) == 0) {
        if (error != nullptr) {
          *error = prefix + " contains unsupported field '" + entry.first + "'";
        }
        return false;
      }
    }

    TextGenerationRecurrentState state;
    std::string kind;
    std::string dataType;
    if (!readString(value, "input", true, &state.input, error) ||
        !readString(value, "output", true, &state.output, error) ||
        !readString(value, "kind", true, &kind, error) ||
        !readString(value, "dataType", true, &dataType, error)) {
      return false;
    }
    if (kind == "GDN") {
      state.kind = TextGenerationRecurrentStateKind::GDN;
    } else if (kind == "CONV") {
      state.kind = TextGenerationRecurrentStateKind::CONV;
    } else {
      if (error != nullptr) *error = prefix + " kind is unsupported: " + kind;
      return false;
    }
    if (!parseCanonicalRecurrentDataType(dataType, &state.dataType)) {
      if (error != nullptr) {
        *error = prefix + " dataType is unsupported: " + dataType;
      }
      return false;
    }

    const JsonValue* shape = member(value, "shape");
    if (shape == nullptr || shape->kind != JsonKind::Array ||
        shape->array.empty()) {
      if (error != nullptr) *error = prefix + " shape must be a non-empty array";
      return false;
    }
    state.shape.reserve(shape->array.size());
    for (const auto& dimension : shape->array) {
      if (dimension.kind != JsonKind::Number ||
          !std::isfinite(dimension.number) ||
          std::floor(dimension.number) != dimension.number ||
          dimension.number <= 0.0 ||
          dimension.number >= 9223372036854775808.0) {
        if (error != nullptr) *error = prefix + " contains an invalid dimension";
        return false;
      }
      state.shape.push_back(static_cast<int64_t>(dimension.number));
    }
    if (!inputs.insert(state.input).second) {
      if (error != nullptr) {
        *error = "duplicate recurrent state input: " + state.input;
      }
      return false;
    }
    if (!outputs.insert(state.output).second) {
      if (error != nullptr) {
        *error = "duplicate recurrent state output: " + state.output;
      }
      return false;
    }
    out->push_back(std::move(state));
  }
  return true;
}

bool validatePositive(const char* field, int32_t value, std::string* error) {
  if (value > 0) return true;
  if (error != nullptr) *error = std::string("metadata field '") + field + "' must be > 0";
  return false;
}

}  // namespace

bool loadTextGenerationMetadata(
    const std::string& path,
    TextGenerationMetadata* out,
    std::string* error) {
  if (out == nullptr) {
    if (error != nullptr) *error = "text-generation metadata output is null";
    return false;
  }
  if (path.empty()) {
    if (error != nullptr) *error = "bundle has no text-generation configPath";
    return false;
  }

  std::ifstream input(path, std::ios::in | std::ios::binary);
  if (!input.good()) {
    if (error != nullptr) *error = "failed to open text-generation metadata: " + path;
    return false;
  }
  const std::string json(
      (std::istreambuf_iterator<char>(input)),
      std::istreambuf_iterator<char>());

  JsonValue root;
  JsonParser parser(json);
  if (!parser.parse(&root, error)) return false;
  if (root.kind != JsonKind::Object) {
    if (error != nullptr) *error = "text-generation metadata root must be an object";
    return false;
  }

  TextGenerationMetadata metadata;
  if (!readInt32(root, "formatVersion", true, &metadata.formatVersion, error)) {
    return false;
  }
  if (metadata.formatVersion != 1 && metadata.formatVersion != 2) {
    if (error != nullptr) {
      *error = "unsupported text-generation formatVersion: " +
               std::to_string(metadata.formatVersion);
    }
    return false;
  }
  if (!readString(root, "profile", true, &metadata.profile, error)) return false;
  const std::string expectedProfile =
      metadata.formatVersion == 1
          ? "causal-lm-in-graph-kv-v1"
          : "causal-lm-in-graph-state-v2";
  if (metadata.profile != expectedProfile) {
    if (error != nullptr) {
      *error = "text-generation formatVersion " +
               std::to_string(metadata.formatVersion) +
               " requires profile '" + expectedProfile + "'";
    }
    return false;
  }

  const JsonValue* io = nullptr;
  if (!requireObject(root, "io", &io, error)) return false;
  if (!readString(*io, "inputIds", true, &metadata.inputIds, error) ||
      !readString(*io, "causalMask", true, &metadata.causalMask, error) ||
      !readString(*io, "positionOffset", true, &metadata.positionOffset, error) ||
      !readString(*io, "cachePosition", true, &metadata.cachePosition, error) ||
      !readString(*io, "actualSequenceLength", true,
                  &metadata.actualSequenceLength, error) ||
      !readString(*io, "logits", true, &metadata.logits, error) ||
      !readString(*io, "prefillLogits", false, &metadata.prefillLogits, error) ||
      !readStringArray(*io, "kvKeyInputs", &metadata.kvKeyInputs, error) ||
      !readStringArray(*io, "kvValueInputs", &metadata.kvValueInputs, error) ||
      !readOptionalKvShapeTemplates(*io, "kvKeyShapes", &metadata.kvKeyShapes, error) ||
      !readOptionalKvShapeTemplates(*io, "kvValueShapes", &metadata.kvValueShapes, error) ||
      !readStringArray(*io, "prefillKeyOutputs",
                       &metadata.prefillKeyOutputs, error) ||
      !readStringArray(*io, "prefillValueOutputs",
                       &metadata.prefillValueOutputs, error)) {
    return false;
  }
  if (metadata.prefillLogits.empty()) metadata.prefillLogits = metadata.logits;
  if (!parseRecurrentStates(
          *io,
          metadata.formatVersion,
          &metadata.recurrentStates,
          error)) {
    return false;
  }

  const size_t layers = metadata.kvKeyInputs.size();
  if (layers == 0 || metadata.kvValueInputs.size() != layers ||
      metadata.prefillKeyOutputs.size() != layers ||
      metadata.prefillValueOutputs.size() != layers) {
    if (error != nullptr) {
      *error = "KV input and prefill output arrays must have the same non-zero layer count";
    }
    return false;
  }
  const bool hasKeyShapes = !metadata.kvKeyShapes.empty();
  const bool hasValueShapes = !metadata.kvValueShapes.empty();
  if (hasKeyShapes != hasValueShapes ||
      (hasKeyShapes && (metadata.kvKeyShapes.size() != layers ||
                        metadata.kvValueShapes.size() != layers))) {
    if (error != nullptr) {
      *error = "KV shape arrays must both be absent or match the KV layer count";
    }
    return false;
  }

  const JsonValue* execution = nullptr;
  if (!requireObject(root, "execution", &execution, error)) return false;
  std::string kvDtype;
  std::string maskDtype;
  if (!readString(*execution, "kvLayout", true, &metadata.kvLayout, error) ||
      !readString(*execution, "kvDtype", true, &kvDtype, error) ||
      !readString(*execution, "maskDtype", true, &maskDtype, error) ||
      !readBoolean(*execution, "planOwnsKvScatter", true,
                   &metadata.planOwnsKvScatter, error)) {
    return false;
  }
  if (upper(metadata.kvLayout) != "BSHD") {
    if (error != nullptr) *error = metadata.profile + " requires BSHD KV layout";
    return false;
  }
  if (!metadata.planOwnsKvScatter) {
    if (error != nullptr) *error = metadata.profile + " requires planOwnsKvScatter=true";
    return false;
  }
  if (!parseDataType(kvDtype, &metadata.kvDataType) ||
      !parseDataType(maskDtype, &metadata.maskDataType)) {
    if (error != nullptr) *error = "unsupported kvDtype or maskDtype in text-generation metadata";
    return false;
  }
  if (metadata.maskDataType != DataType::FLOAT32 &&
      metadata.maskDataType != DataType::HALF &&
      metadata.maskDataType != DataType::BFLOAT16) {
    if (error != nullptr) *error = "maskDtype must be FLOAT32, FLOAT16, or BFLOAT16";
    return false;
  }

  const JsonValue* limits = nullptr;
  if (!requireObject(root, "limits", &limits, error) ||
      !readInt32(*limits, "contextLength", true,
                 &metadata.contextLength, error) ||
      !readInt32(*limits, "maxPrefillLength", true,
                 &metadata.maxPrefillLength, error) ||
      !readInt32(*limits, "maxBatchSize", false,
                 &metadata.maxBatchSize, error)) {
    return false;
  }
  if (!validatePositive("limits.contextLength", metadata.contextLength, error) ||
      !validatePositive("limits.maxPrefillLength",
                        metadata.maxPrefillLength, error) ||
      metadata.maxPrefillLength >= metadata.contextLength) {
    if (error != nullptr && metadata.maxPrefillLength >= metadata.contextLength) {
      *error = "maxPrefillLength must be smaller than contextLength";
    }
    return false;
  }
  if (metadata.maxBatchSize != 1) {
    if (error != nullptr) {
      *error = metadata.profile + " supports maxBatchSize=1 only";
    }
    return false;
  }

  const JsonValue* tokens = nullptr;
  if (!requireObject(root, "tokens", &tokens, error) ||
      !readInt64(*tokens, "bosId", false, &metadata.bosId, error) ||
      !readInt64(*tokens, "padId", true, &metadata.padId, error) ||
      !readInt64(*tokens, "unkId", false, &metadata.unkId, error) ||
      !readIntArray(*tokens, "eosIds", &metadata.eosIds, error)) {
    return false;
  }
  if (metadata.padId < 0) {
    if (error != nullptr) *error = "tokens.padId must be >= 0";
    return false;
  }

  if (const JsonValue* sampling = member(root, "samplingDefaults")) {
    if (sampling->kind != JsonKind::Object) {
      if (error != nullptr) *error = "samplingDefaults must be an object";
      return false;
    }
    if (!readInt32(*sampling, "maxNewTokens", false,
                   &metadata.sampling.maxNewTokens, error) ||
        !readInt32(*sampling, "minNewTokens", false,
                   &metadata.sampling.minNewTokens, error) ||
        !readNumber(*sampling, "temperature", false,
                    &metadata.sampling.temperature, error) ||
        !readInt32(*sampling, "topK", false,
                   &metadata.sampling.topK, error) ||
        !readNumber(*sampling, "topP", false,
                    &metadata.sampling.topP, error) ||
        !readNumber(*sampling, "minP", false,
                    &metadata.sampling.minP, error) ||
        !readNumber(*sampling, "repetitionPenalty", false,
                    &metadata.sampling.repetitionPenalty, error) ||
        !readNumber(*sampling, "frequencyPenalty", false,
                    &metadata.sampling.frequencyPenalty, error) ||
        !readNumber(*sampling, "presencePenalty", false,
                    &metadata.sampling.presencePenalty, error) ||
        !readNumber(*sampling, "typicalP", false,
                    &metadata.sampling.typicalP, error) ||
        !readNumber(*sampling, "xtcProbability", false,
                    &metadata.sampling.xtcProbability, error) ||
        !readNumber(*sampling, "xtcThreshold", false,
                    &metadata.sampling.xtcThreshold, error) ||
        !readInt64(*sampling, "seed", false,
                   &metadata.sampling.seed, error)) {
      return false;
    }
  }

  if (!readString(root, "chatTemplatePath", false,
                  &metadata.chatTemplatePath, error)) {
    return false;
  }

  if (metadata.sampling.maxNewTokens <= 0 ||
      metadata.sampling.minNewTokens < 0 ||
      metadata.sampling.topK < 0 ||
      metadata.sampling.temperature < 0.0 ||
      metadata.sampling.topP < 0.0 || metadata.sampling.topP > 1.0 ||
      metadata.sampling.minP < 0.0 || metadata.sampling.minP > 1.0 ||
      metadata.sampling.repetitionPenalty <= 0.0 ||
      metadata.sampling.typicalP <= 0.0 || metadata.sampling.typicalP > 1.0 ||
      metadata.sampling.xtcProbability < 0.0 ||
      metadata.sampling.xtcProbability > 1.0 ||
      metadata.sampling.xtcThreshold < 0.0 ||
      metadata.sampling.xtcThreshold > 1.0) {
    if (error != nullptr) *error = "samplingDefaults contains an out-of-range value";
    return false;
  }

  *out = std::move(metadata);
  if (error != nullptr) error->clear();
  return true;
}

}  // namespace detail
}  // namespace runtime
}  // namespace dsp
}  // namespace sd
