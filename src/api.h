#ifndef API_H
#define API_H

#if defined(_WIN32) && !defined(__MINGW32__)
    #define DART_API __declspec(dllimport)
#else
    #define DART_API __attribute__ ((visibility ("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void dart_output(const char *buffer);

DART_API char * llama_default_params(void);

DART_API int llama_llm_init(char * params);

DART_API int llama_prompt(char * messages, dart_output * output);

DART_API void llama_llm_stop(void);

DART_API void llama_llm_free(void);

// New embedding functions
DART_API float * llama_get_embedding(char * text, int * embedding_size);

DART_API void llama_free_embedding(float * embedding);

#ifdef __cplusplus
}
#endif

#endif