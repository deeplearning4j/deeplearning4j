#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <stdexcept>

namespace tokenizers {

/**
 * @brief Result structure for tokenization operations
 */
struct TokenizeResult {
    std::vector<std::string> tokens;
    std::vector<uint32_t> ids;
    std::vector<std::pair<size_t, size_t>> offsets;
    bool success = false;
    std::string error_message;
    
    TokenizeResult() = default;
    TokenizeResult(const TokenizeResult&) = default;
    TokenizeResult& operator=(const TokenizeResult&) = default;
    TokenizeResult(TokenizeResult&&) = default;
    TokenizeResult& operator=(TokenizeResult&&) = default;
};

/**
 * @brief High-level C++ wrapper for HuggingFace tokenizers
 * 
 * This class provides a safe, RAII-compliant interface to the underlying
 * Rust tokenizer implementation via FFI.
 */
class TokenizerWrapper {
public:
    /**
     * @brief Construct a tokenizer from a file path
     * @param model_path Path to the tokenizer configuration file (JSON)
     * @throws std::runtime_error if the tokenizer cannot be loaded
     */
    explicit TokenizerWrapper(const std::string& model_path);
    
    /**
     * @brief Construct a tokenizer from JSON string
     * @param json_config JSON configuration string
     * @throws std::runtime_error if the JSON is invalid
     */
    static TokenizerWrapper from_json(const std::string& json_config);
    
    /**
     * @brief Move constructor
     */
    TokenizerWrapper(TokenizerWrapper&& other) noexcept;
    
    /**
     * @brief Move assignment operator
     */
    TokenizerWrapper& operator=(TokenizerWrapper&& other) noexcept;
    
    /**
     * @brief Destructor
     */
    ~TokenizerWrapper();
    
    // Disable copy operations for safety
    TokenizerWrapper(const TokenizerWrapper&) = delete;
    TokenizerWrapper& operator=(const TokenizerWrapper&) = delete;
    
    /**
     * @brief Encode text into tokens
     * @param text Input text to tokenize
     * @param add_special_tokens Whether to add special tokens (e.g., [CLS], [SEP])
     * @return TokenizeResult containing tokens, IDs, and offsets
     */
    TokenizeResult encode(const std::string& text, bool add_special_tokens = true);
    
    /**
     * @brief Decode token IDs back to text
     * @param ids Vector of token IDs
     * @param skip_special_tokens Whether to skip special tokens in output
     * @return Decoded text string
     */
    std::string decode(const std::vector<uint32_t>& ids, bool skip_special_tokens = true);
    
    /**
     * @brief Encode multiple texts in batch
     * @param texts Vector of input texts
     * @param add_special_tokens Whether to add special tokens
     * @return Vector of TokenizeResult for each input
     */
    std::vector<TokenizeResult> encode_batch(const std::vector<std::string>& texts, 
                                           bool add_special_tokens = true);
    
    /**
     * @brief Get the vocabulary size of the tokenizer
     * @return Number of tokens in vocabulary
     */
    size_t get_vocab_size() const;
    
    /**
     * @brief Check if the tokenizer is valid and ready to use
     * @return true if valid, false otherwise
     */
    bool is_valid() const;
    
    /**
     * @brief Get the underlying handle (for advanced usage)
     * @return Pointer to internal handle (do not free manually)
     */
    void* get_handle() const;

private:
    /**
     * @brief Private constructor for internal use (e.g., from_json)
     */
    TokenizerWrapper();
    
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * @brief Exception thrown by tokenizer operations
 */
class TokenizerException : public std::runtime_error {
public:
    explicit TokenizerException(const std::string& message) 
        : std::runtime_error(message) {}
};

} // namespace tokenizers
