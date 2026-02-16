// Simple circular linked list for fixed-size samples.
#include "circular_list.h"

#include <stdlib.h>

bool circular_list_init(CircularList *list, size_t size) {
    if (!list || size == 0) return false;

    list->head = NULL;
    list->current = NULL;
    list->size = 0;
    list->sum_sq = 0.0f;

    CircularNode *first = (CircularNode *)malloc(sizeof(CircularNode));
    if (!first) return false;
    first->value = 0.0f;
    first->next = first;

    CircularNode *prev = first;
    for (size_t i = 1; i < size; i++) {
        CircularNode *node = (CircularNode *)malloc(sizeof(CircularNode));
        if (!node) {
            list->head = first;
            list->current = first;
            list->size = i;
            circular_list_free(list);
            return false;
        }
        node->value = 0.0f;
        node->next = first;
        prev->next = node;
        prev = node;
    }

    list->head = first;
    list->current = first;
    list->size = size;
    list->sum_sq = 0.0f;
    return true;
}

bool circular_list_add(CircularList *list, float value, float *overwritten) {
    if (!list || !list->current || list->size == 0) return false;

    float old = list->current->value;
    if (overwritten) {
        *overwritten = old;
    }
    list->current->value = value;
    list->sum_sq -= old * old;
    list->sum_sq += value * value;
    list->current = list->current->next;
    return true;
}

float circular_list_sum_sq(const CircularList *list) {
    if (!list) return 0.0f;
    return list->sum_sq;
}

void circular_list_free(CircularList *list) {
    if (!list || !list->head || list->size == 0) return;

    CircularNode *node = list->head;
    for (size_t i = 0; i < list->size; i++) {
        CircularNode *next = node->next;
        free(node);
        node = next;
    }

    list->head = NULL;
    list->current = NULL;
    list->size = 0;
    list->sum_sq = 0.0f;
}
