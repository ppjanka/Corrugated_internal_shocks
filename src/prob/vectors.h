#include "copyright.h"
/*============================================================================*/
/*! \file vectors.h
 *  \brief Reimplementation of the vector class in C.
 *  Example usage (type template based on https://stackoverflow.com/questions/10950828/simulation-of-templates-in-c-for-a-queue-data-type):
 *   #define TYPE int
     #define TYPED_NAME(x) int_##x
     #include "queue_impl.h"
     #undef TYPE
     #undef TYPED_NAME
 *  Author: Patryk Pjanka, Nordita, 2021 */
/*============================================================================*/

#include <math.h>

#if false // use for debugging
  #define TYPE int
  #define TYPED_NAME(x) int_##x
#endif

// This is ugly, but it is C I guess...
typedef struct TYPED_NAME(vector_element) {
  TYPE first;
  TYPE second;
  struct TYPED_NAME(vector_element)* previous;
  struct TYPED_NAME(vector_element)* next;
};
struct TYPED_NAME(vector_element) TYPED_NAME(vector_element_default) = {0,0,NULL,NULL};

typedef struct TYPED_NAME(vector) {
  struct TYPED_NAME(vector_element)* first_element;
  struct TYPED_NAME(vector_element)* last_element;
  int n_elements;
};
struct TYPED_NAME(vector) TYPED_NAME(vector_default) = {NULL,NULL,0};

void TYPED_NAME(append_to_vector) (struct TYPED_NAME(vector) *vector, TYPE first, TYPE second) {
  // create the new element
  struct TYPED_NAME(vector_element)* element = (struct TYPED_NAME(vector_element)*) malloc (sizeof(struct TYPED_NAME(vector_element)));
  element->first = first;
  element->second = second;
  element->next = NULL;
  // append to vector
  if (vector->first_element != NULL) {
    element->previous = vector->last_element;
    vector->last_element->next = element;
    vector->last_element = element;
  } else { //initialize empty vector with first element
    element->previous = NULL;
    vector->first_element = element;
    vector->last_element = element;
  }
  vector->n_elements++;
}

void TYPED_NAME(drop_from_vector) (struct TYPED_NAME(vector) *vector) {
  struct TYPED_NAME(vector_element)* element = vector->last_element;
  vector->last_element = element->previous;
  if (vector->last_element == NULL) {
    vector->first_element = NULL;
  }
  free(element);
  vector->n_elements--;
}

void TYPED_NAME(clear_vector) (struct TYPED_NAME(vector) *vector) {
  while (vector->last_element != NULL) {
    TYPED_NAME(drop_from_vector) (vector);
  }
}

TYPE** TYPED_NAME(vector_to_array) (struct TYPED_NAME(vector) *vector) {
  TYPE** result = (TYPE**) malloc(vector->n_elements * sizeof(TYPE*));
  struct TYPED_NAME(vector_element)* element = vector->first_element;
  for (int i = vector->n_elements-1; i >= 0; i--) {
    result[i] = (TYPE*) malloc(2*sizeof(TYPE));
    result[i][0] = vector->last_element->first;
    result[i][1] = vector->last_element->second;
    TYPED_NAME(drop_from_vector)(vector);
  }
  return result;
}

void TYPED_NAME(copy_vector) (struct TYPED_NAME(vector) *src, struct TYPED_NAME(vector) *dest) {
  if (dest->n_elements > 0) TYPED_NAME(clear_vector) (dest);
  struct TYPED_NAME(vector_element)* element = src->first_element;
  for (int i = 0; i < src->n_elements; i++) {
    TYPED_NAME(append_to_vector) (dest, element->first, element->second);
    element = element->next;
  }
}
