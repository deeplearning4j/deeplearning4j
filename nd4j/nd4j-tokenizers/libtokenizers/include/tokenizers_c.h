#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle for tokenizer instance
 */
typedef struct OpaqueTokenizer OpaqueTokenizer;

/**
 * @brief Opaque handle for encoding result
 */
typedef struct OpaqueEncoding OpaqueEncoding;

/**
 * @brief Opaque handle for model manager
 */
typedef struct OpaqueModelManager OpaqueModelManager;

/**
 * @brief Error codes for tokenizer operations
 */
typedef enum {
    TOKENIZER_SUCCESS = 0,
    TOKENIZER_ERROR_INVALID_INPUT = 1,
    TOKENIZER_ERROR_FILE_NOT_FOUND = 2,
    TOKENIZER_ERROR_INVALID_JSON = 3,
    TOKENIZER_ERROR_MEMORY_ALLOCATION = 4,
    TOKENIZER_ERROR_UNKNOWN = 5
} TokenizerError;

/**
 * @brief Result structure for operations that can fail
 */
typedef struct {
    TokenizerError error_code;
    const char* error_message;
} TokenizerResult;

// =============================================================================
// Core Tokenizer Functions
// =============================================================================

/**
 * @brief Create a tokenizer from a file path
 * @param model_path Path to the tokenizer configuration file
 * @return Pointer to tokenizer instance, or NULL on failure
 */
OpaqueTokenizer* create_tokenizer_from_file(const char* model_path);

/**
 * @brief Create a tokenizer from JSON string
 * @param json_config JSON configuration string
 * @return Pointer to tokenizer instance, or NULL on failure
 */
OpaqueTokenizer* create_tokenizer_from_json(const char* json_config);

/**
 * @brief Free a tokenizer instance
 * @param tokenizer Tokenizer instance to free
 */
void free_tokenizer(OpaqueTokenizer* tokenizer);

/**
 * @brief Check if a tokenizer is valid
 * @param tokenizer Tokenizer instance
 * @return true if valid, false otherwise
 */
bool tokenizer_is_valid(OpaqueTokenizer* tokenizer);

/**
 * @brief Get vocabulary size
 * @param tokenizer Tokenizer instance
 * @return Vocabulary size, or 0 if invalid
 */
size_t get_vocab_size(OpaqueTokenizer* tokenizer);

// =============================================================================
// Encoding Functions
// =============================================================================

/**
 * @brief Encode text into tokens
 * @param tokenizer Tokenizer instance
 * @param text Input text to encode
 * @param add_special_tokens Whether to add special tokens
 * @return Encoding result, or NULL on failure
 */
OpaqueEncoding* encode_text(OpaqueTokenizer* tokenizer, const char* text, bool add_special_tokens);

/**
 * @brief Encode multiple texts in batch
 * @param tokenizer Tokenizer instance
 * @param texts Array of input texts
 * @param num_texts Number of texts in array
 * @param add_special_tokens Whether to add special tokens
 * @return Array of encoding results, or NULL on failure
 */
OpaqueEncoding** encode_batch(OpaqueTokenizer* tokenizer, const char** texts, size_t num_texts, bool add_special_tokens);

/**
 * @brief Free an encoding result
 * @param encoding Encoding to free
 */
void free_encoding(OpaqueEncoding* encoding);

/**
 * @brief Free a batch of encoding results
 * @param encodings Array of encodings to free
 * @param num_encodings Number of encodings in array
 */
void free_encoding_batch(OpaqueEncoding** encodings, size_t num_encodings);

// =============================================================================
// Encoding Result Access Functions
// =============================================================================

/**
 * @brief Get the number of tokens in an encoding
 * @param encoding Encoding result
 * @return Number of tokens, or 0 if invalid
 */
size_t encoding_get_length(OpaqueEncoding* encoding);

/**
 * @brief Get token IDs from an encoding
 * @param encoding Encoding result
 * @return Array of token IDs, or NULL if invalid
 */
const uint32_t* encoding_get_ids(OpaqueEncoding* encoding);

/**
 * @brief Get token strings from an encoding
 * @param encoding Encoding result
 * @return Array of token strings, or NULL if invalid
 */
const char** encoding_get_tokens(OpaqueEncoding* encoding);

/**
 * @brief Get token offsets from an encoding
 * @param encoding Encoding result
 * @param start_offsets Output array for start offsets
 * @param end_offsets Output array for end offsets
 * @return Number of offsets, or 0 if invalid
 */
size_t encoding_get_offsets(OpaqueEncoding* encoding, size_t** start_offsets, size_t** end_offsets);

// =============================================================================
// Decoding Functions
// =============================================================================

/**
 * @brief Decode token IDs back to text
 * @param tokenizer Tokenizer instance
 * @param ids Array of token IDs
 * @param num_ids Number of IDs in array
 * @param skip_special_tokens Whether to skip special tokens
 * @return Decoded text string, or NULL on failure (must be freed with free_string)
 */
char* decode_ids(OpaqueTokenizer* tokenizer, const uint32_t* ids, size_t num_ids, bool skip_special_tokens);

/**
 * @brief Free a string returned by decode operations
 * @param str String to free
 */
void free_string(char* str);

// =============================================================================
// Model Manager Functions
// =============================================================================

/**
 * @brief Create a model manager instance
 * @return Pointer to model manager, or NULL on failure
 */
OpaqueModelManager* create_model_manager();

/**
 * @brief Free a model manager instance
 * @param manager Model manager to free
 */
void free_model_manager(OpaqueModelManager* manager);

/**
 * @brief Check if a model file is valid
 * @param model_path Path to model file
 * @return true if valid, false otherwise
 */
bool is_valid_model_file(const char* model_path);

/**
 * @brief Get list of available embedded models
 * @param num_models Output parameter for number of models
 * @return Array of model names, or NULL if none available
 */
const char** get_embedded_models(size_t* num_models);

/**
 * @brief Free embedded models list
 * @param models Array returned by get_embedded_models
 * @param num_models Number of models in array
 */
void free_embedded_models(const char** models, size_t num_models);

// =============================================================================
// Error Handling Functions
// =============================================================================

/**
 * @brief Get the last error that occurred
 * @return Error result structure
 */
TokenizerResult get_last_error();

/**
 * @brief Clear the last error
 */
void clear_last_error();

/**
 * @brief Get version information
 * @return Version string (do not free)
 */
const char* get_tokenizer_version();

/**
 * @brief Get build information
 * @return Build info string (do not free)
 */
const char* get_build_info();

#ifdef __cplusplus
}
#endif
