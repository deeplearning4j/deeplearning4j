#include "tokenizer_wrapper.h"
#include "tokenizers_ffi.h"
#include <stdexcept>
#include <cstring>

namespace tokenizers {

// Implementation using Rust FFI
class TokenizerWrapper::Impl {
public:
    TokenizerHandle* handle = nullptr;
    bool valid = false;

    Impl(const std::string& path) {
        handle = ffi_tokenizer_from_file(path.c_str());
        valid = (handle != nullptr);
        if (!valid) {
            const char* err = ffi_tokenizer_get_last_error();
            throw TokenizerException(err ? err : "Failed to load tokenizer from file");
        }
    }

    Impl(const std::string& json, bool is_json) {
        (void)is_json; // Mark as intentionally unused
        handle = ffi_tokenizer_from_json(json.c_str());
        valid = (handle != nullptr);
        if (!valid) {
            const char* err = ffi_tokenizer_get_last_error();
            throw TokenizerException(err ? err : "Failed to create tokenizer from JSON");
        }
    }

    ~Impl() {
        if (handle) {
            ffi_tokenizer_free(handle);
            handle = nullptr;
        }
    }

    TokenizeResult encode(const std::string& text, bool add_special_tokens) {
        TokenizeResult result;

        if (!valid || !handle) {
            result.success = false;
            result.error_message = "Tokenizer is not valid";
            return result;
        }

        EncodingHandle* encoding = ffi_tokenizer_encode(handle, text.c_str(), add_special_tokens);
        if (!encoding) {
            result.success = false;
            const char* err = ffi_tokenizer_get_last_error();
            result.error_message = err ? err : "Encoding failed";
            return result;
        }

        // Get encoding data
        size_t length = ffi_encoding_get_length(encoding);
        const uint32_t* ids = ffi_encoding_get_ids(encoding);
        const char* const* tokens = ffi_encoding_get_tokens(encoding);

        // Copy to result
        result.ids.assign(ids, ids + length);
        for (size_t i = 0; i < length; ++i) {
            result.tokens.push_back(tokens[i] ? tokens[i] : "");
            // Offsets not available in simple FFI - use placeholder
            result.offsets.push_back({0, 0});
        }

        ffi_encoding_free(encoding);
        result.success = true;
        return result;
    }

    std::string decode(const std::vector<uint32_t>& ids, bool skip_special_tokens) {
        if (!valid || !handle || ids.empty()) {
            return "";
        }

        char* decoded = ffi_tokenizer_decode(handle, ids.data(), ids.size(), skip_special_tokens);
        if (!decoded) {
            return "";
        }

        std::string result(decoded);
        ffi_tokenizer_free_string(decoded);
        return result;
    }

    std::vector<TokenizeResult> encode_batch(const std::vector<std::string>& texts, bool add_special_tokens) {
        std::vector<TokenizeResult> results;
        for (const auto& text : texts) {
            results.push_back(encode(text, add_special_tokens));
        }
        return results;
    }

    size_t get_vocab_size() const {
        if (!valid || !handle) {
            return 0;
        }
        return ffi_tokenizer_get_vocab_size(handle);
    }
};

// TokenizerWrapper implementation
TokenizerWrapper::TokenizerWrapper(const std::string& model_path)
    : pImpl(std::make_unique<Impl>(model_path)) {
}

TokenizerWrapper TokenizerWrapper::from_json(const std::string& json_config) {
    TokenizerWrapper wrapper;
    wrapper.pImpl = std::make_unique<Impl>(json_config, true);
    return wrapper;
}

TokenizerWrapper::TokenizerWrapper(TokenizerWrapper&& other) noexcept
    : pImpl(std::move(other.pImpl)) {
}

TokenizerWrapper& TokenizerWrapper::operator=(TokenizerWrapper&& other) noexcept {
    if (this != &other) {
        pImpl = std::move(other.pImpl);
    }
    return *this;
}

TokenizerWrapper::~TokenizerWrapper() = default;

TokenizerWrapper::TokenizerWrapper()
    : pImpl(nullptr) {
}

TokenizeResult TokenizerWrapper::encode(const std::string& text, bool add_special_tokens) {
    if (!pImpl) {
        TokenizeResult result;
        result.success = false;
        result.error_message = "Tokenizer not initialized";
        return result;
    }
    return pImpl->encode(text, add_special_tokens);
}

std::string TokenizerWrapper::decode(const std::vector<uint32_t>& ids, bool skip_special_tokens) {
    if (!pImpl) {
        return "";
    }
    return pImpl->decode(ids, skip_special_tokens);
}

std::vector<TokenizeResult> TokenizerWrapper::encode_batch(const std::vector<std::string>& texts, bool add_special_tokens) {
    if (!pImpl) {
        return {};
    }
    return pImpl->encode_batch(texts, add_special_tokens);
}

size_t TokenizerWrapper::get_vocab_size() const {
    if (!pImpl) {
        return 0;
    }
    return pImpl->get_vocab_size();
}

bool TokenizerWrapper::is_valid() const {
    return pImpl && pImpl->valid;
}

void* TokenizerWrapper::get_handle() const {
    return pImpl ? pImpl->handle : nullptr;
}

} // namespace tokenizers
