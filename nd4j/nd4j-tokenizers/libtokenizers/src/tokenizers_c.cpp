#include "tokenizers_c.h"
#include "tokenizer_wrapper.h"
#include "model_manager.h"
#include <cstring>
#include <memory>
#include <thread>
#include <iostream>

// Thread-local error storage
thread_local TokenizerResult last_error = {TOKENIZER_SUCCESS, nullptr};

// Helper function to set error
static void set_error(TokenizerError code, const char* message) {
    last_error.error_code = code;
    last_error.error_message = message;
}

// Helper function to clear error
static void clear_error() {
    last_error.error_code = TOKENIZER_SUCCESS;
    last_error.error_message = nullptr;
}

// =============================================================================
// Core Tokenizer Functions
// =============================================================================

extern "C" {

OpaqueTokenizer* create_tokenizer_from_file(const char* model_path) {
    if (!model_path) {
        set_error(TOKENIZER_ERROR_INVALID_INPUT, "Model path cannot be null");
        return nullptr;
    }
    
    try {
        clear_error();
        auto tokenizer = new tokenizers::TokenizerWrapper(model_path);
        return reinterpret_cast<OpaqueTokenizer*>(tokenizer);
    } catch (const std::exception& e) {
        set_error(TOKENIZER_ERROR_UNKNOWN, e.what());
        return nullptr;
    }
}

OpaqueTokenizer* create_tokenizer_from_json(const char* json_config) {
    if (!json_config) {
        set_error(TOKENIZER_ERROR_INVALID_INPUT, "JSON config cannot be null");
        return nullptr;
    }
    
    try {
        clear_error();
        auto tokenizer = new tokenizers::TokenizerWrapper(tokenizers::TokenizerWrapper::from_json(json_config));
        return reinterpret_cast<OpaqueTokenizer*>(tokenizer);
    } catch (const std::exception& e) {
        set_error(TOKENIZER_ERROR_INVALID_JSON, e.what());
        return nullptr;
    }
}

void free_tokenizer(OpaqueTokenizer* tokenizer) {
    if (tokenizer) {
        delete reinterpret_cast<tokenizers::TokenizerWrapper*>(tokenizer);
    }
}

bool tokenizer_is_valid(OpaqueTokenizer* tokenizer) {
    if (!tokenizer) {
        return false;
    }
    
    try {
        auto* wrapper = reinterpret_cast<tokenizers::TokenizerWrapper*>(tokenizer);
        return wrapper->is_valid();
    } catch (...) {
        return false;
    }
}

size_t get_vocab_size(OpaqueTokenizer* tokenizer) {
    if (!tokenizer) {
        set_error(TOKENIZER_ERROR_INVALID_INPUT, "Tokenizer cannot be null");
        return 0;
    }
    
    try {
        clear_error();
        auto* wrapper = reinterpret_cast<tokenizers::TokenizerWrapper*>(tokenizer);
        return wrapper->get_vocab_size();
    } catch (const std::exception& e) {
        set_error(TOKENIZER_ERROR_UNKNOWN, e.what());
        return 0;
    }
}

// =============================================================================
// Encoding Functions
// =============================================================================

// Simple wrapper for encoding results
struct EncodingResult {
    std::vector<std::string> tokens;
    std::vector<uint32_t> ids;
    std::vector<std::pair<size_t, size_t>> offsets;
    std::vector<const char*> token_ptrs;  // For C string array
    std::vector<size_t> start_offsets;
    std::vector<size_t> end_offsets;
    
    void update_token_ptrs() {
        token_ptrs.clear();
        for (const auto& token : tokens) {
            token_ptrs.push_back(token.c_str());
        }
    }
    
    void update_offsets() {
        start_offsets.clear();
        end_offsets.clear();
        for (const auto& offset : offsets) {
            start_offsets.push_back(offset.first);
            end_offsets.push_back(offset.second);
        }
    }
};

OpaqueEncoding* encode_text(OpaqueTokenizer* tokenizer, const char* text, bool add_special_tokens) {
    if (!tokenizer || !text) {
        set_error(TOKENIZER_ERROR_INVALID_INPUT, "Tokenizer and text cannot be null");
        return nullptr;
    }
    
    try {
        clear_error();
        auto* wrapper = reinterpret_cast<tokenizers::TokenizerWrapper*>(tokenizer);
        auto result = wrapper->encode(text, add_special_tokens);
        
        if (!result.success) {
            set_error(TOKENIZER_ERROR_UNKNOWN, result.error_message.c_str());
            return nullptr;
        }
        
        auto* encoding = new EncodingResult();
        encoding->tokens = std::move(result.tokens);
        encoding->ids = std::move(result.ids);
        encoding->offsets = std::move(result.offsets);
        encoding->update_token_ptrs();
        encoding->update_offsets();
        
        return reinterpret_cast<OpaqueEncoding*>(encoding);
    } catch (const std::exception& e) {
        set_error(TOKENIZER_ERROR_UNKNOWN, e.what());
        return nullptr;
    }
}

OpaqueEncoding** encode_batch(OpaqueTokenizer* tokenizer, const char** texts, size_t num_texts, bool add_special_tokens) {
    if (!tokenizer || !texts || num_texts == 0) {
        set_error(TOKENIZER_ERROR_INVALID_INPUT, "Invalid batch encoding parameters");
        return nullptr;
    }
    
    try {
        clear_error();
        auto* wrapper = reinterpret_cast<tokenizers::TokenizerWrapper*>(tokenizer);
        
        std::vector<std::string> input_texts;
        for (size_t i = 0; i < num_texts; ++i) {
            if (texts[i]) {
                input_texts.emplace_back(texts[i]);
            }
        }
        
        auto results = wrapper->encode_batch(input_texts, add_special_tokens);
        
        auto** encodings = new OpaqueEncoding*[num_texts];
        for (size_t i = 0; i < num_texts; ++i) {
            if (i < results.size() && results[i].success) {
                auto* encoding = new EncodingResult();
                encoding->tokens = std::move(results[i].tokens);
                encoding->ids = std::move(results[i].ids);
                encoding->offsets = std::move(results[i].offsets);
                encoding->update_token_ptrs();
                encoding->update_offsets();
                encodings[i] = reinterpret_cast<OpaqueEncoding*>(encoding);
            } else {
                encodings[i] = nullptr;
            }
        }
        
        return encodings;
    } catch (const std::exception& e) {
        set_error(TOKENIZER_ERROR_UNKNOWN, e.what());
        return nullptr;
    }
}

void free_encoding(OpaqueEncoding* encoding) {
    if (encoding) {
        delete reinterpret_cast<EncodingResult*>(encoding);
    }
}

void free_encoding_batch(OpaqueEncoding** encodings, size_t num_encodings) {
    if (encodings) {
        for (size_t i = 0; i < num_encodings; ++i) {
            free_encoding(encodings[i]);
        }
        delete[] encodings;
    }
}

// =============================================================================
// Encoding Result Access Functions
// =============================================================================

size_t encoding_get_length(OpaqueEncoding* encoding) {
    if (!encoding) {
        return 0;
    }
    
    auto* result = reinterpret_cast<EncodingResult*>(encoding);
    return result->ids.size();
}

const uint32_t* encoding_get_ids(OpaqueEncoding* encoding) {
    if (!encoding) {
        return nullptr;
    }
    
    auto* result = reinterpret_cast<EncodingResult*>(encoding);
    return result->ids.data();
}

const char** encoding_get_tokens(OpaqueEncoding* encoding) {
    if (!encoding) {
        return nullptr;
    }
    
    auto* result = reinterpret_cast<EncodingResult*>(encoding);
    return result->token_ptrs.data();
}

size_t encoding_get_offsets(OpaqueEncoding* encoding, size_t** start_offsets, size_t** end_offsets) {
    if (!encoding || !start_offsets || !end_offsets) {
        return 0;
    }
    
    auto* result = reinterpret_cast<EncodingResult*>(encoding);
    *start_offsets = result->start_offsets.data();
    *end_offsets = result->end_offsets.data();
    return result->offsets.size();
}

// =============================================================================
// Decoding Functions
// =============================================================================

char* decode_ids(OpaqueTokenizer* tokenizer, const uint32_t* ids, size_t num_ids, bool skip_special_tokens) {
    if (!tokenizer || !ids || num_ids == 0) {
        set_error(TOKENIZER_ERROR_INVALID_INPUT, "Invalid decode parameters");
        return nullptr;
    }
    
    try {
        clear_error();
        auto* wrapper = reinterpret_cast<tokenizers::TokenizerWrapper*>(tokenizer);
        
        std::vector<uint32_t> id_vector(ids, ids + num_ids);
        std::string decoded = wrapper->decode(id_vector, skip_special_tokens);
        
        // Allocate and copy string
        char* result = new char[decoded.length() + 1];
        std::strcpy(result, decoded.c_str());
        return result;
    } catch (const std::exception& e) {
        set_error(TOKENIZER_ERROR_UNKNOWN, e.what());
        return nullptr;
    }
}

void free_string(char* str) {
    if (str) {
        delete[] str;
    }
}

// =============================================================================
// Model Manager Functions
// =============================================================================

OpaqueModelManager* create_model_manager() {
    // Model manager is static, just return a dummy pointer
    return reinterpret_cast<OpaqueModelManager*>(0x1);
}

void free_model_manager(OpaqueModelManager* manager) {
    // Nothing to free for static manager
}

bool is_valid_model_file(const char* model_path) {
    if (!model_path) {
        return false;
    }
    
    try {
        return tokenizers::ModelManager::is_valid_model_file(model_path);
    } catch (...) {
        return false;
    }
}

const char** get_embedded_models(size_t* num_models) {
    if (!num_models) {
        return nullptr;
    }
    
    try {
        auto models = tokenizers::ModelManager::get_embedded_models();
        *num_models = models.size();
        
        if (models.empty()) {
            return nullptr;
        }
        
        // For now, return empty since embedded models aren't implemented
        *num_models = 0;
        return nullptr;
    } catch (...) {
        *num_models = 0;
        return nullptr;
    }
}

void free_embedded_models(const char** models, size_t num_models) {
    // Nothing to free for now since we return nullptr
}

// =============================================================================
// Error Handling Functions
// =============================================================================

TokenizerResult get_last_error() {
    return last_error;
}

void clear_last_error() {
    clear_error();
}

const char* get_tokenizer_version() {
    return "1.0.0-SNAPSHOT";
}

const char* get_build_info() {
    return "tokenizers-rust C++ wrapper build";
}

} // extern "C"
