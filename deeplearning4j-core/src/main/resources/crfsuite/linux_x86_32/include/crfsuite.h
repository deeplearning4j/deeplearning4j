/*
 *      CRFsuite library.
 *
 * Copyright (c) 2007-2010, Naoaki Okazaki
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the names of the authors nor the names of its contributors
 *       may be used to endorse or promote products derived from this
 *       software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* $Id$ */

#ifndef    __CRFSUITE_H__
#define    __CRFSUITE_H__

#ifdef    __cplusplus
extern "C" {
#endif/*__cplusplus*/

#include <stdio.h>
#include <stdarg.h>

/** 
 * \addtogroup crfsuite_api CRFSuite C API
 * @{
 *
 *  The CRFSuite C API provides a low-level library for manupulating
 *  CRFSuite in C language.
 */

/** 
 * \addtogroup crfsuite_misc Miscellaneous definitions and functions
 * @{
 */

/** Version number of CRFSuite library. */
#define CRFSUITE_VERSION    "0.12"

/** Copyright string of CRFSuite library. */
#define CRFSUITE_COPYRIGHT  "Copyright (c) 2007-2011 Naoaki Okazaki"

/** Type of a float value. */
typedef double floatval_t;

/** Maximum value of a float value. */
#define    FLOAT_MAX    DBL_MAX

/**
 * Status codes.
 */
enum {
    /** Success. */
    CRFSUITE_SUCCESS = 0,
    /** Unknown error occurred. */
    CRFSUITEERR_UNKNOWN = 0x80000000,
    /** Insufficient memory. */
    CRFSUITEERR_OUTOFMEMORY,
    /** Unsupported operation. */
    CRFSUITEERR_NOTSUPPORTED,
    /** Incompatible data. */
    CRFSUITEERR_INCOMPATIBLE,
    /** Internal error. */
    CRFSUITEERR_INTERNAL_LOGIC,
    /** Overflow. */
    CRFSUITEERR_OVERFLOW,
    /** Not implemented. */
    CRFSUITEERR_NOTIMPLEMENTED,
};

/**@}*/



/**
 * \addtogroup crfsuite_object Object interfaces and utilities.
 * @{
 */

struct tag_crfsuite_model;
/** CRFSuite model interface. */
typedef struct tag_crfsuite_model crfsuite_model_t;

struct tag_crfsuite_trainer;
/** CRFSuite trainer interface. */
typedef struct tag_crfsuite_trainer crfsuite_trainer_t;

struct tag_crfsuite_tagger;
/** CRFSuite tagger interface. */
typedef struct tag_crfsuite_tagger crfsuite_tagger_t;

struct tag_crfsuite_dictionary;
/** CRFSuite dictionary interface. */
typedef struct tag_crfsuite_dictionary crfsuite_dictionary_t;

struct tag_crfsuite_params;
/** CRFSuite parameter interface. */
typedef struct tag_crfsuite_params crfsuite_params_t;

/**@}*/



/**
 * \addtogroup crfsuite_data Dataset (attribute, item, instance, dataset)
 * @{
 */

/**
 * An attribute.
 *  An attribute consists of an attribute id with its value.
 */
typedef struct {
    int         aid;                /**< Attribute id. */
    floatval_t  value;              /**< Value of the attribute. */
} crfsuite_attribute_t;

/**
 * An item.
 *  An item consists of an array of attributes.
 */
typedef struct {
    /** Number of contents associated with the item. */
    int             num_contents;
    /** Maximum number of contents (internal use). */
    int             cap_contents;
    /** Array of the attributes. */
    crfsuite_attribute_t    *contents;
} crfsuite_item_t;

/**
 * An instance (sequence of items and labels).
 *  An instance consists of a sequence of items and labels.
 */
typedef struct {
    /** Number of items/labels in the sequence. */
    int         num_items;
    /** Maximum number of items/labels (internal use). */
    int         cap_items;
    /** Array of the item sequence. */
    crfsuite_item_t  *items;
    /** Array of the label sequence. */
    int         *labels;
    /** Group ID of the instance. */
	int         group;
} crfsuite_instance_t;

/**
 * A data set.
 *  A data set consists of an array of instances and dictionary objects
 *  for attributes and labels.
 */
typedef struct {
    /** Number of instances. */
    int                 num_instances;
    /** Maximum number of instances (internal use). */
    int                 cap_instances;
    /** Array of instances. */
    crfsuite_instance_t*     instances;

    /** Dictionary object for attributes. */
    crfsuite_dictionary_t    *attrs;
    /** Dictionary object for labels. */
    crfsuite_dictionary_t    *labels;
} crfsuite_data_t;

/**@}*/



/**
 * \addtogroup crfsuite_evaluation Evaluation utility
 * @{
 */

/**
 * Label-wise performance values.
 */
typedef struct {
    /** Number of correct predictions. */
    int         num_correct;
    /** Number of occurrences of the label in the gold-standard data. */
    int         num_observation;
    /** Number of predictions. */
    int         num_model;
    /** Precision. */
    floatval_t  precision;
    /** Recall. */
    floatval_t  recall;
    /** F1 score. */
    floatval_t  fmeasure;
} crfsuite_label_evaluation_t;

/**
 * An overall performance values.
 */
typedef struct {
    /** Number of labels. */
    int         num_labels;
    /** Array of label-wise evaluations. */
    crfsuite_label_evaluation_t* tbl;

    /** Number of correctly predicted items. */
    int         item_total_correct;
    /** Total number of items. */
    int         item_total_num;
    /** Total number of occurrences of labels in the gold-standard data. */
    int         item_total_observation;
    /** Total number of predictions. */
    int         item_total_model;
    /** Item-level accuracy. */
    floatval_t  item_accuracy;

    /** Number of correctly predicted instances. */
    int         inst_total_correct;
    /** Total number of instances. */
    int         inst_total_num;
    /** Instance-level accuracy. */
    floatval_t  inst_accuracy;

    /** Macro-averaged precision. */
    floatval_t  macro_precision;
    /** Macro-averaged recall. */
    floatval_t  macro_recall;
    /** Macro-averaged F1 score. */
    floatval_t  macro_fmeasure;
} crfsuite_evaluation_t;

/**@}*/


/**
 * \addtogroup crfsuite_object
 * @{
 */

/**
 * Type of callback function for logging.
 *  @param  user        Pointer to the user-defined data.
 *  @param  format      Format string (compatible with prinf()).
 *  @param  args        Optional arguments for the format string.
 *  @return int         \c 0 to continue; non-zero to cancel the training.
 */
typedef int (*crfsuite_logging_callback)(void *user, const char *format, va_list args);


/**
 * CRFSuite model interface.
 */
struct tag_crfsuite_model {
    /**
     * Pointer to the internal data (internal use only).
     */
    void *internal;
    
    /**
     * Reference counter (internal use only).
     */
    int nref;

    /**
     * Increment the reference counter.
     *  @param  model       The pointer to this model instance.
     *  @return int         The reference count after this increment.
     */
    int (*addref)(crfsuite_model_t* model);

    /**
     * Decrement the reference counter.
     *  @param  model       The pointer to this model instance.
     *  @return int         The reference count after this operation.
     */
    int (*release)(crfsuite_model_t* model);

    /**
     * Obtain the pointer to crfsuite_tagger_t interface.
     *  @param  model       The pointer to this model instance.
     *  @param  ptr_tagger  The pointer that receives a crfsuite_tagger_t
     *                      pointer.
     *  @return int         The status code.
     */
    int (*get_tagger)(crfsuite_model_t* model, crfsuite_tagger_t** ptr_tagger);

    /**
     * Obtain the pointer to crfsuite_dictionary_t interface for labels.
     *  @param  model       The pointer to this model instance.
     *  @param  ptr_labels  The pointer that receives a crfsuite_dictionary_t
     *                      pointer.
     *  @return int         The status code.
     */
    int (*get_labels)(crfsuite_model_t* model, crfsuite_dictionary_t** ptr_labels);

    /**
     * Obtain the pointer to crfsuite_dictionary_t interface for attributes.
     *  @param  model       The pointer to this model instance.
     *  @param  ptr_attrs   The pointer that receives a crfsuite_dictionary_t
     *                      pointer.
     *  @return int         The status code.
     */
    int (*get_attrs)(crfsuite_model_t* model, crfsuite_dictionary_t** ptr_attrs);

    /**
     * Print the model in human-readable format.
     *  @param  model       The pointer to this model instance.
     *  @param  fpo         The FILE* pointer.
     *  @param  ptr_attrs   The pointer that receives a crfsuite_dictionary_t
     *                      pointer.
     *  @return int         The status code.
     */
    int (*dump)(crfsuite_model_t* model, FILE *fpo);
};



/**
 * CRFSuite trainer interface.
 */
struct tag_crfsuite_trainer {
    /**
     * Pointer to the internal data (internal use only).
     */
    void *internal;
    
    /**
     * Reference counter (internal use only).
     */
    int nref;

    /**
     * Increment the reference counter.
     *  @param  trainer     The pointer to this trainer instance.
     *  @return int         The reference count after this increment.
     */
    int (*addref)(crfsuite_trainer_t* trainer);

    /**
     * Decrement the reference counter.
     *  @param  trainer     The pointer to this trainer instance.
     *  @return int         The reference count after this operation.
     */
    int (*release)(crfsuite_trainer_t* trainer);

    /**
     * Obtain the pointer to crfsuite_params_t interface.
     *  @param  trainer     The pointer to this trainer instance.
     *  @return crfsuite_params_t*  The pointer to crfsuite_params_t.
     */
    crfsuite_params_t* (*params)(crfsuite_trainer_t* trainer);

    /**
     * Set the callback function and user-defined data.
     *  @param  trainer     The pointer to this trainer instance.
     *  @param  user        The pointer to the user-defined data.
     *  @param  cbm         The pointer to the callback function.
     */
    void (*set_message_callback)(crfsuite_trainer_t* trainer, void *user, crfsuite_logging_callback cbm);

    /**
     * Start a training process.
     *  @param  trainer     The pointer to this trainer instance.
     *  @param  data        The poiinter to the data set.
     *  @param  filename    The filename to which the trainer stores the model.
     *                      If an empty string is specified, this function
     *                      does not sture the model to a file.
     *  @param  holdout     The holdout group.
     *  @return int         The status code.
     */
    int (*train)(crfsuite_trainer_t* trainer, const crfsuite_data_t *data, const char *filename, int holdout);
};

/**
 * CRFSuite tagger interface.
 */
struct tag_crfsuite_tagger {
    /**
     * Pointer to the internal data (internal use only).
     */
    void *internal;

    /**
     * Reference counter (internal use only).
     */
    int nref;

    /**
     * Increment the reference counter.
     *  @param  tagger      The pointer to this tagger instance.
     *  @return int         The reference count after this increment.
     */
    int (*addref)(crfsuite_tagger_t* tagger);

    /**
     * Decrement the reference counter.
     *  @param  tagger      The pointer to this tagger instance.
     *  @return int         The reference count after this operation.
     */
    int (*release)(crfsuite_tagger_t* tagger);

    /**
     * Set an instance to the tagger.
     *  @param  tagger      The pointer to this tagger instance.
     *  @param  inst        The item sequence to be tagged.
     *  @return int         The status code.
     */
    int (*set)(crfsuite_tagger_t* tagger, crfsuite_instance_t *inst);

    /**
     * Obtain the number of items in the current instance.
     *  @param  tagger      The pointer to this tagger instance.
     *  @return int         The number of items of the instance set by
     *                      set() function.
     *  @return int         The status code.
     */
    int (*length)(crfsuite_tagger_t* tagger);

    /**
     * Find the Viterbi label sequence.
     *  @param  tagger      The pointer to this tagger instance.
     *  @param  labels      The label array that receives the Viterbi label
     *                      sequence. The number of elements in the array must
     *                      be no smaller than the number of item.
     *  @param  ptr_score   The pointer to a float variable that receives the
     *                      score of the Viterbi label sequence.
     *  @return int         The status code.
     */
    int (*viterbi)(crfsuite_tagger_t* tagger, int *labels, floatval_t *ptr_score);

    /**
     * Compute the score of a label sequence.
     *  @param  tagger      The pointer to this tagger instance.
     *  @param  path        The label sequence.
     *  @param  ptr_score   The pointer to a float variable that receives the
     *                      score of the label sequence.
     *  @return int         The status code.
     */
    int (*score)(crfsuite_tagger_t* tagger, int *path, floatval_t *ptr_score);

    /**
     * Compute the log of the partition factor (normalization constant).
     *  @param  tagger      The pointer to this tagger instance.
     *  @param  ptr_score   The pointer to a float variable that receives the
     *                      logarithm of the partition factor.
     *  @return int         The status code.
     */
    int (*lognorm)(crfsuite_tagger_t* tagger, floatval_t *ptr_norm);

    /**
     * Compute the marginal probability of a label at a position.
     *  This function computes P(y_t = l | x), the probability when
     *  y_t is the label (l).
     *  @param  tagger      The pointer to this tagger instance.
     *  @param  l           The label.
     *  @param  t           The position.
     *  @param  ptr_prob    The pointer to a float variable that receives the
     *                      marginal probability.
     *  @return int         The status code.
     */
    int (*marginal_point)(crfsuite_tagger_t *tagger, int l, int t, floatval_t *ptr_prob);

    /**
     * Compute the marginal probability of a partial label sequence.
     *  @param  tagger      The pointer to this tagger instance.
     *  @param  path        The partial label sequence.
     *  @param  begin       The start position of the partial label sequence.
     *  @param  end         The last+1 position of the partial label sequence.
     *  @param  ptr_prob    The pointer to a float variable that receives the
     *                      marginal probability.
     *  @return int         The status code.
     */
    int (*marginal_path)(crfsuite_tagger_t *tagger, const int *path, int begin, int end, floatval_t *ptr_prob);
};

/**
 * CRFSuite dictionary interface.
 */
struct tag_crfsuite_dictionary {
    /**
     * Pointer to the internal data (internal use only).
     */
    void *internal;

    /**
     * Reference counter (internal use only).
     */
    int nref;

    /**
     * Increment the reference counter.
     *  @param  dic         The pointer to this dictionary instance.
     *  @return int         The reference count after this increment.
     */
    int (*addref)(crfsuite_dictionary_t* dic);

    /**
     * Decrement the reference counter.
     *  @param  dic         The pointer to this dictionary instance.
     *  @return int         The reference count after this operation.
     */
    int (*release)(crfsuite_dictionary_t* dic);

    /**
     * Assign and obtain the integer ID for the string.
     *  @param  dic         The pointer to this dictionary instance.
     *  @param  str         The string.
     *  @return int         The ID associated with the string if any,
     *                      the new ID otherwise.
     */
    int (*get)(crfsuite_dictionary_t* dic, const char *str);

    /**
     * Obtain the integer ID for the string.
     *  @param  dic         The pointer to this dictionary instance.
     *  @param  str         The string.
     *  @return int         The ID associated with the string if any,
     *                      \c -1 otherwise.
     */
    int (*to_id)(crfsuite_dictionary_t* dic, const char *str);

    /**
     * Obtain the string for the ID.
     *  @param  dic         The pointer to this dictionary instance.
     *  @param  id          the string ID.
     *  @param  pstr        \c *pstr points to the string associated with
     *                      the ID if any, \c NULL otherwise.
     *  @return int         \c 0 if the string ID is associated with a string,
     *                      \c 1 otherwise.
     */
    int (*to_string)(crfsuite_dictionary_t* dic, int id, char const **pstr);

    /**
     * Obtain the number of strings in the dictionary.
     *  @param  dic         The pointer to this dictionary instance.
     *  @return int         The number of strings stored in the dictionary.
     */
    int (*num)(crfsuite_dictionary_t* dic);

    /**
     * Free the memory block allocated by to_string() function.
     *  @param  dic         The pointer to this dictionary instance.
     *  @param  str         The pointer to the string whose memory block is
     *                      freed.
     */
    void (*free)(crfsuite_dictionary_t* dic, const char *str);
};

/**
 * CRFSuite parameter interface.
 */
struct tag_crfsuite_params {
    /**
     * Pointer to the instance data (internal use only).
     */
    void *internal;

    /**
     * Reference counter (internal use only).
     */
    int nref;

    /**
     * Increment the reference counter.
     *  @param  params      The pointer to this parameter instance.
     *  @return int         The reference count after this increment.
     */
    int (*addref)(crfsuite_params_t* params);

    /**
     * Decrement the reference counter.
     *  @param  params      The pointer to this parameter instance.
     *  @return int         The reference count after this operation.
     */
    int (*release)(crfsuite_params_t* params);

    /**
     * Obtain the number of available parameters.
     *  @param  params      The pointer to this parameter instance.
     *  @return int         The number of parameters maintained by this object.
     */
    int (*num)(crfsuite_params_t* params);

    /**
     * Obtain the name of a parameter.
     *  @param  params      The pointer to this parameter instance.
     *  @param  i           The parameter index.
     *  @param  ptr_name    *ptr_name points to the parameter name.
     *  @return int         \c 0 always.
     */
    int (*name)(crfsuite_params_t* params, int i, char **ptr_name);

    /**
     * Set a parameter value.
     *  @param  params      The pointer to this parameter instance.
     *  @param  name        The parameter name.
     *  @param  value       The parameter value in string format.
     *  @return int         \c 0 if the parameter is found, \c -1 otherwise.
     */
    int (*set)(crfsuite_params_t* params, const char *name, const char *value);

    /**
     * Get a parameter value.
     *  @param  params      The pointer to this parameter instance.
     *  @param  name        The parameter name.
     *  @param  ptr_value   *ptr_value presents the parameter value in string
     *                      format.
     *  @return int         \c 0 if the parameter is found, \c -1 otherwise.
     */
    int (*get)(crfsuite_params_t* params, const char *name, char **ptr_value);

    /**
     * Set an integer value of a parameter.
     *  @param  params      The pointer to this parameter instance.
     *  @param  name        The parameter name.
     *  @param  value       The parameter value.
     *  @return int         \c 0 if the parameter value is set successfully,
     *                      \c -1 otherwise (unknown parameter or incompatible
     *                      type).
     */
    int (*set_int)(crfsuite_params_t* params, const char *name, int value);

    /**
     * Set a float value of a parameter.
     *  @param  params      The pointer to this parameter instance.
     *  @param  name        The parameter name.
     *  @param  value       The parameter value.
     *  @return int         \c 0 if the parameter value is set successfully,
     *                      \c -1 otherwise (unknown parameter or incompatible
     *                      type).
     */
    int (*set_float)(crfsuite_params_t* params, const char *name, floatval_t value);

    /**
     * Set a string value of a parameter.
     *  @param  params      The pointer to this parameter instance.
     *  @param  name        The parameter name.
     *  @param  value       The parameter value.
     *  @return int         \c 0 if the parameter value is set successfully,
     *                      \c -1 otherwise (unknown parameter or incompatible
     *                      type).
     */
    int (*set_string)(crfsuite_params_t* params, const char *name, const char *value);

    /**
     * Get an integer value of a parameter.
     *  @param  params      The pointer to this parameter instance.
     *  @param  name        The parameter name.
     *  @param  ptr_value   The pointer to a variable that receives the
     *                      integer value.
     *  @return int         \c 0 if the parameter value is obtained
     *                      successfully, \c -1 otherwise (unknown parameter
     *                      or incompatible type).
     */
    int (*get_int)(crfsuite_params_t* params, const char *name, int *ptr_value);

    /**
     * Get a float value of a parameter.
     *  @param  params      The pointer to this parameter instance.
     *  @param  name        The parameter name.
     *  @param  ptr_value   The pointer to a variable that receives the
     *                      float value.
     *  @return int         \c 0 if the parameter value is obtained
     *                      successfully, \c -1 otherwise (unknown parameter
     *                      or incompatible type).
     */
    int (*get_float)(crfsuite_params_t* params, const char *name, floatval_t *ptr_value);

    /**
     * Get a string value of a parameter.
     *  @param  params      The pointer to this parameter instance.
     *  @param  name        The parameter name.
     *  @param  ptr_value   *ptr_value presents the parameter value.
     *  @return int         \c 0 if the parameter value is obtained
     *                      successfully, \c -1 otherwise (unknown parameter
     *                      or incompatible type).
     */
    int (*get_string)(crfsuite_params_t* params, const char *name, char **ptr_value);

    /**
     * Get the help message of a parameter.
     *  @param  params      The pointer to this parameter instance.
     *  @param  name        The parameter name.
     *  @param  ptr_type    The pointer to \c char* to which this function
     *                      store the type of the parameter.
     *  @param  ptr_help    The pointer to \c char* to which this function
     *                      store the help message of the parameter.
     *  @return int         \c 0 if the parameter is found, \c -1 otherwise.
     */
    int (*help)(crfsuite_params_t* params, const char *name, char **ptr_type, char **ptr_help);

    /**
     * Free the memory block of a string allocated by this object.
     *  @param  params      The pointer to this parameter instance.
     *  @param  str         The pointer to the string.
     */
    void (*free)(crfsuite_params_t* params, const char *str);
};

/**@}*/



/**
 * \addtogroup crfsuite_object
 * @{
 */

/**
 * Create an instance of an object by an interface identifier.
 *  @param  iid         The interface identifier.
 *  @param  ptr         The pointer to \c void* that points to the
 *                      instance of the object if successful,
 *                      *ptr points to \c NULL otherwise.
 *  @return int         \c 0 if this function creates an object successfully,
 *                      \c 1 otherwise.
 */
int crfsuite_create_instance(const char *iid, void **ptr);

/**
 * Create an instance of a model object from a model file.
 *  @param  filename    The filename of the model.
 *  @param  ptr         The pointer to \c void* that points to the
 *                      instance of the model object if successful,
 *                      *ptr points to \c NULL otherwise.
 *  @return int         \c 0 if this function creates an object successfully,
 *                      \c 1 otherwise.
 */
int crfsuite_create_instance_from_file(const char *filename, void **ptr);

/**
 * Create instances of tagging object from a model file.
 *  @param  filename    The filename of the model.
 *  @param  ptr_tagger  The pointer to \c void* that points to the
 *                      instance of the tagger object if successful,
 *                      *ptr points to \c NULL otherwise.
 *  @param  ptr_attrs   The pointer to \c void* that points to the
 *                      instance of the dictionary object for attributes
 *                      if successful, *ptr points to \c NULL otherwise.
 *  @param  ptr_labels  The pointer to \c void* that points to the
 *                      instance of the dictionary object for labels
 *                      if successful, *ptr points to \c NULL otherwise.
 *  @return int         \c 0 if this function creates an object successfully,
 *                      \c 1 otherwise.
 */
int crfsuite_create_tagger(
    const char *filename,
    crfsuite_tagger_t** ptr_tagger,
    crfsuite_dictionary_t** ptr_attrs,
    crfsuite_dictionary_t** ptr_labels
    );

/**@}*/



/**
 * \addtogroup crfsuite_data
 * @{
 */

/**
 * Initialize an attribute structure.
 *  @param  attr        The pointer to crfsuite_attribute_t.
 */
void crfsuite_attribute_init(crfsuite_attribute_t* attr);

/**
 * Set an attribute and its value.
 *  @param  attr        The pointer to crfsuite_attribute_t.
 *  @param  aid         The attribute identifier.
 *  @param  value       The attribute value.
 */
void crfsuite_attribute_set(crfsuite_attribute_t* attr, int aid, floatval_t value);

/**
 * Copy the content of an attribute structure.
 *  @param  dst         The pointer to the destination.
 *  @param  src         The pointer to the source.
 */
void crfsuite_attribute_copy(crfsuite_attribute_t* dst, const crfsuite_attribute_t* src);

/**
 * Swap the contents of two attribute structures.
 *  @param  x           The pointer to an attribute structure.
 *  @param  y           The pointer to another attribute structure.
 */
void crfsuite_attribute_swap(crfsuite_attribute_t* x, crfsuite_attribute_t* y);

/**
 * Initialize an item structure.
 *  @param  item        The pointer to crfsuite_item_t.
 */
void crfsuite_item_init(crfsuite_item_t* item);

/**
 * Initialize an item structure with the number of attributes.
 *  @param  item        The pointer to crfsuite_item_t.
 *  @param  num_attributes  The number of attributes.
 */
void crfsuite_item_init_n(crfsuite_item_t* item, int num_attributes);

/**
 * Uninitialize an item structure.
 *  @param  item        The pointer to crfsuite_item_t.
 */
void crfsuite_item_finish(crfsuite_item_t* item);

/**
 * Copy the content of an item structure.
 *  @param  dst         The pointer to the destination.
 *  @param  src         The pointer to the source.
 */
void crfsuite_item_copy(crfsuite_item_t* dst, const crfsuite_item_t* src);

/**
 * Swap the contents of two item structures.
 *  @param  x           The pointer to an item structure.
 *  @param  y           The pointer to another item structure.
 */
void crfsuite_item_swap(crfsuite_item_t* x, crfsuite_item_t* y);

/**
 * Append an attribute to the item structure.
 *  @param  item        The pointer to crfsuite_item_t.
 *  @param  attr        The attribute to be added to the item.
 *  @return int         \c 0 if successful, \c -1 otherwise.
 */
int  crfsuite_item_append_attribute(crfsuite_item_t* item, const crfsuite_attribute_t* attr);

/**
 * Check whether the item has no attribute.
 *  @param  item        The pointer to crfsuite_item_t.
 *  @return int         \c 1 if the item has no attribute, \c 0 otherwise.
 */
int  crfsuite_item_empty(crfsuite_item_t* item);



/**
 * Initialize an instance structure.
 *  @param  seq         The pointer to crfsuite_instance_t.
 */
void crfsuite_instance_init(crfsuite_instance_t* seq);

/**
 * Initialize an instance structure with the number of items.
 *  @param  seq         The pointer to crfsuite_instance_t.
 *  @param  num_items   The number of items.
 */
void crfsuite_instance_init_n(crfsuite_instance_t* seq, int num_items);

/**
 * Uninitialize an instance structure.
 *  @param  seq         The pointer to crfsuite_instance_t.
 */
void crfsuite_instance_finish(crfsuite_instance_t* seq);

/**
 * Copy the content of an instance structure.
 *  @param  dst         The pointer to the destination.
 *  @param  src         The pointer to the source.
 */
void crfsuite_instance_copy(crfsuite_instance_t* dst, const crfsuite_instance_t* src);

/**
 * Swap the contents of two instance structures.
 *  @param  x           The pointer to an instance structure.
 *  @param  y           The pointer to another instance structure.
 */
void crfsuite_instance_swap(crfsuite_instance_t* x, crfsuite_instance_t* y);

/**
 * Append a pair of item and label to the instance structure.
 *  @param  seq         The pointer to crfsuite_instance_t.
 *  @param  item        The item to be added to the instance.
 *  @param  label       The label to be added to the instance.
 *  @return int         \c 0 if successful, \c -1 otherwise.
 */
int  crfsuite_instance_append(crfsuite_instance_t* seq, const crfsuite_item_t* item, int label);

/**
 * Check whether the instance has no item.
 *  @param  seq         The pointer to crfsuite_instance_t.
 *  @return int         \c 1 if the instance has no attribute, \c 0 otherwise.
 */
int  crfsuite_instance_empty(crfsuite_instance_t* seq);



/**
 * Initialize a dataset structure.
 *  @param  data        The pointer to crfsuite_data_t.
 */
void crfsuite_data_init(crfsuite_data_t* data);

/**
 * Initialize a dataset structure with the number of instances.
 *  @param  data        The pointer to crfsuite_data_t.
 *  @param  n           The number of instances.
 */
void crfsuite_data_init_n(crfsuite_data_t* data, int n);

/**
 * Uninitialize a dataset structure.
 *  @param  data        The pointer to crfsuite_data_t.
 */
void crfsuite_data_finish(crfsuite_data_t* data);

/**
 * Copy the content of a dataset structure.
 *  @param  dst         The pointer to the destination.
 *  @param  src         The pointer to the source.
 */
void crfsuite_data_copy(crfsuite_data_t* dst, const crfsuite_data_t* src);

/**
 * Swap the contents of two dataset structures.
 *  @param  x           The pointer to a dataset structure.
 *  @param  y           The pointer to another dataset structure.
 */
void crfsuite_data_swap(crfsuite_data_t* x, crfsuite_data_t* y);

/**
 * Append an instance to the dataset structure.
 *  @param  data        The pointer to crfsuite_data_t.
 *  @param  inst        The instance to be added to the dataset.
 *  @return int         \c 0 if successful, \c -1 otherwise.
 */
int  crfsuite_data_append(crfsuite_data_t* data, const crfsuite_instance_t* inst);

/**
 * Obtain the maximum length of the instances in the dataset.
 *  @param  data        The pointer to crfsuite_data_t.
 *  @return int         The maximum number of items of the instances in the
 *                      dataset.
 */
int  crfsuite_data_maxlength(crfsuite_data_t* data);

/**
 * Obtain the total number of items in the dataset.
 *  @param  data        The pointer to crfsuite_data_t.
 *  @return int         The total number of items in the dataset.
 */
int  crfsuite_data_totalitems(crfsuite_data_t* data);

/**@}*/

/**
 * \addtogroup crfsuite_evaluation
 */
/**@{*/

/**
 * Initialize an evaluation structure.
 *  @param  eval        The pointer to crfsuite_evaluation_t.
 *  @param  n           The number of labels in the dataset.
 */
void crfsuite_evaluation_init(crfsuite_evaluation_t* eval, int n);

/**
 * Uninitialize an evaluation structure.
 *  @param  eval        The pointer to crfsuite_evaluation_t.
 */
void crfsuite_evaluation_finish(crfsuite_evaluation_t* eval);

/**
 * Reset an evaluation structure.
 *  @param  eval        The pointer to crfsuite_evaluation_t.
 */
void crfsuite_evaluation_clear(crfsuite_evaluation_t* eval);

/**
 * Accmulate the correctness of the predicted label sequence.
 *  @param  eval        The pointer to crfsuite_evaluation_t.
 *  @param  reference   The reference label sequence.
 *  @param  prediction  The predicted label sequence.
 *  @param  T           The length of the label sequence.
 *  @return int         \c 0 if succeeded, \c 1 otherwise.
 */
int crfsuite_evaluation_accmulate(crfsuite_evaluation_t* eval, const int* reference, const int* prediction, int T);

/**
 * Finalize the evaluation result.
 *  @param  eval        The pointer to crfsuite_evaluation_t.
 */
void crfsuite_evaluation_finalize(crfsuite_evaluation_t* eval);

/**
 * Print the evaluation result.
 *  @param  eval        The pointer to crfsuite_evaluation_t.
 *  @param  labels      The pointer to the label dictionary.
 *  @param  cbm         The callback function to receive the evaluation result.
 *  @param  user        The pointer to the user data that is forwarded to the
 *                      callback function.
 */
void crfsuite_evaluation_output(crfsuite_evaluation_t* eval, crfsuite_dictionary_t* labels, crfsuite_logging_callback cbm, void *user);

/**@}*/



/** 
 * \addtogroup crfsuite_misc Miscellaneous definitions and functions
 * @{
 */

/**
 * Increments the value of the integer variable as an atomic operation.
 *  @param  count       The pointer to the integer variable.
 *  @return             The value after this increment.
 */
int crfsuite_interlocked_increment(int *count);

/**
 * Decrements the value of the integer variable as an atomic operation.
 *  @param  count       The pointer to the integer variable.
 *  @return             The value after this decrement.
 */
int crfsuite_interlocked_decrement(int *count);

/**@}*/

/**@}*/

/**
@mainpage CRFsuite: a fast implementation of Conditional Random Fields (CRFs)

@section intro Introduction

This document describes information for using
<a href="http://www.chokkan.org/software/crfsuite">CRFsuite</a> from external
programs. CRFsuite provides two APIs:
- @link crfsuite_api C API @endlink: low-level and complete interface, which
  is used by the official frontend program.
- @link crfsuite_hpp_api C++/SWIG API @endlink: high-level and easy-to-use
  interface for a number of programming languages (e.g, C++ and Python),
  which is a wrapper for the C API.

*/


#ifdef    __cplusplus
}
#endif/*__cplusplus*/

#endif/*__CRFSUITE_H__*/
