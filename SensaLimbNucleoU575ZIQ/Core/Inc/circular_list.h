// Simple circular linked list for fixed-size samples.
#ifndef CIRCULAR_LIST_H
#define CIRCULAR_LIST_H

#include <stdbool.h>
#include <stddef.h>

typedef struct CircularNode {
    float value;
    struct CircularNode *next;
} CircularNode;

typedef struct {
    CircularNode *head;
    CircularNode *current;
    size_t size;
    float sum_sq; // running sum of squares
} CircularList;

// Allocates a circular list with "size" nodes.
bool circular_list_init(CircularList *list, size_t size);

// Writes a new value at the current position, returns overwritten value.
bool circular_list_add(CircularList *list, float value, float *overwritten);

// Returns the running sum of squares.
float circular_list_sum_sq(const CircularList *list);

// Frees all nodes and resets list fields.
void circular_list_free(CircularList *list);

#endif
