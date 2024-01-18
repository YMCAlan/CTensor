#pragma once
#ifndef MACRO_H
#define MACRO_H

#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition, message) \
  do { \
    if (!(condition)) { \
      fprintf(stderr, "CHECK failed:  %s:%d - %s\n", __FILE__, __LINE__, message); \
      abort(); \
    } \
  } while (0)


#define ASSERT(condition) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "Assertion failed: %s at %s:%d\n", #condition, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_RANGE(value, lower, upper) \
    do { \
        if ((value) < (lower) || (value) >= (upper)) { \
            fprintf(stderr, "Value out of range at %s:%d\n", __FILE__, __LINE__); \
            abort(); \
        } \
    } while(0)

#define MALLOC(TYPE, SIZE) ((TYPE*)safeMalloc(sizeof(TYPE) * SIZE));
#define REALLOC(PTR, TYPE, SIZE) ((TYPE*)safeRealloc(PTR,sizeof(TYPE) * SIZE))

void* safeMalloc(size_t size);
void* safeRealloc(void* ptr, size_t size);

#define COPY_ARRAY(source, destination, size, type) \
    do { \
        static_assert(sizeof(source) == sizeof(destination), "Source and destination arrays must have the same size"); \
        memcpy((destination), (source), (size) * sizeof(type)); \
    } while (0) 

#define SET_ARRAY(source, size, value) do { \
    for (size_t i = 0; i < (size); i++) { \
        (source)[i] = (value); \
    } \
} while (0)

#define PRINT_ARRAY(arr, size, format) do { \
    printf("[ "); \
    for (size_t i = 0; i < (size); i++) { \
        printf(format, (arr)[i]); \
        if (i < (size) - 1) { \
            printf(", "); \
        } \
    } \
    printf(" ]\n"); \
} while (0)

#endif // !MACRO_H
