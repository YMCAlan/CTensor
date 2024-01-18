#include "macro.h"

void* safeMalloc(size_t size)
{
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Memory allocation failed at %s:%d\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void* safeRealloc(void* ptr, size_t size)
{
    void* newPtr = realloc(ptr, size);
    if (!newPtr) {
        fprintf(stderr, "Memory reallocation failed at %s:%d\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    return newPtr;
}