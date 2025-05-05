part of 'package:llama_sdk/llama_sdk.dart';

enum LlamaModelType {
  chat,
  embedding
}

/// A class that isolates the Llama implementation to run in a separate isolate.
///
/// This class implements the [Llama] interface and provides methods to interact
/// with the Llama model in an isolated environment.
///
/// The [LlamaIsolated] constructor initializes the isolate with the provided
/// model, context, and sampling parameters.
///
/// The [prompt] method sends a list of [LlamaMessage] to the isolate and returns
/// a stream of responses. It waits for the isolate to be initialized before
/// sending the messages.
///
/// The [stop] method sends a signal to the isolate to stop processing. It waits
/// for the isolate to be initialized before sending the signal.
///
/// The [reload] method stops the current operation and reloads the isolate.
class Llama {
  Completer _initialized = Completer();
  StreamController<String> _responseController = StreamController<String>()..close();
  Isolate? _isolate;
  SendPort? _sendPort;
  ReceivePort? _receivePort;

  LlamaController _controller;
  final LlamaModelType modelType;

  /// Gets the current LlamaController instance.
  ///
  /// The [LlamaController] instance contains the parameters used by the llama.
  ///
  /// Returns the current [LlamaController] instance.
  LlamaController get controller => _controller;

  set controller(LlamaController value) {
    _controller = value;
    stop();
  }

  /// Constructs an instance of [Llama].
  ///
  /// Initializes the [Llama] with the provided parameters and sets up
  /// the listener.
  ///
  /// Parameters:
  /// - [controller]: The parameters required for the Llama model.
  /// - [modelType]: The type of model (chat or embedding).
  Llama(LlamaController controller, {this.modelType = LlamaModelType.chat}) : _controller = controller {
    controller.addListener(() {
      // Reload the model when controller parameters change
      reload();
    });
  }

  void _listener() async {
    _receivePort = ReceivePort();

    final workerParams = _LlamaWorkerParams(
      sendPort: _receivePort!.sendPort,
      controller: _controller,
    );

    _isolate = await Isolate.spawn(_LlamaWorker.entry, workerParams.toRecord());

    await for (final data in _receivePort!) {
      if (data is SendPort) {
        _sendPort = data;
        _initialized.complete();
      } else if (data is String) {
        _responseController.add(data);
      } else if (data == null) {
        _responseController.close();
      }
    }
  }

  /// Generates a stream of responses based on the provided list of chat messages.
  ///
  /// This method takes a list of [LlamaMessage] objects and returns a [Stream] of
  /// strings, where each string represents a response generated from the chat messages.
  ///
  /// The stream allows for asynchronous processing of the chat messages, enabling
  /// real-time or batched responses.
  ///
  /// - Parameter messages: A list of [LlamaMessage] objects that represent the chat history.
  /// - Returns: A [Stream] of strings, where each string is a generated response.
  Stream<String> prompt(List<LlamaMessage> messages) async* {
    if (!_initialized.isCompleted) {
      _listener();
      await _initialized.future;
    }

    _responseController = StreamController<String>();

    _sendPort!.send(messages.toRecords());

    await for (final response in _responseController.stream) {
      yield response;
    }
  }

  /// Stops the current operation or process.
  ///
  /// This method should be called to terminate any ongoing tasks or
  /// processes that need to be halted. It ensures that resources are
  /// properly released and the system is left in a stable state.
  void stop() => lib.llama_llm_stop();

  /// Frees the resources used by the Llama model.
  void reload() {
    lib.llama_llm_free();
    _isolate?.kill(priority: Isolate.immediate);
    _receivePort?.close();
    _initialized = Completer();
  }

  /// Gets the embedding vector for the given text.
  /// 
  /// This method generates an embedding vector for the input text using the
  /// loaded LLaMA model. The embedding can be used for semantic similarity
  /// comparisons, document retrieval, and other vector-based operations.
  /// 
  /// Parameters:
  /// - [text]: The input text to generate embeddings for.
  /// 
  /// Returns a [Future<List<double>>] containing the embedding vector.
  /// The size of the vector depends on the model's embedding dimensions.
  Future<List<double>> getEmbedding(String text) async {
    if (modelType != LlamaModelType.embedding && !_controller.embeddings) {
      throw Exception('This model is not configured for embeddings. Please set embeddings=true or use an embedding model.');
    }

    if (!_initialized.isCompleted) {
      _listener();
      await _initialized.future;
    }

    final textPtr = text.toNativeUtf8();
    final embeddingSizePtr = calloc<Int>();

    try {
      final embeddingPtr = lib.llama_get_embedding(textPtr, embeddingSizePtr);
      if (embeddingPtr == nullptr) {
        throw Exception('Failed to generate embedding');
      }

      final embeddingSize = embeddingSizePtr.value;
      final embedding = embeddingPtr.asTypedList(embeddingSize);
      final result = List<double>.from(embedding);

      lib.llama_free_embedding(embeddingPtr);
      return result;
    } finally {
      calloc.free(textPtr);
      calloc.free(embeddingSizePtr);
    }
  }

  /// Computes the cosine similarity between two embedding vectors.
  /// 
  /// Parameters:
  /// - [embedding1]: First embedding vector
  /// - [embedding2]: Second embedding vector
  /// 
  /// Returns a value between -1 and 1, where 1 means most similar.
  double cosineSimilarity(List<double> embedding1, List<double> embedding2) {
    if (embedding1.length != embedding2.length) {
      throw ArgumentError('Embeddings must have the same dimensions');
    }

    double dotProduct = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;

    for (int i = 0; i < embedding1.length; i++) {
      dotProduct += embedding1[i] * embedding2[i];
      norm1 += embedding1[i] * embedding1[i];
      norm2 += embedding2[i] * embedding2[i];
    }

    norm1 = sqrt(norm1);
    norm2 = sqrt(norm2);

    if (norm1 == 0 || norm2 == 0) {
      return 0.0;
    }

    return dotProduct / (norm1 * norm2);
  }

  /// Stream chat completions with relevant context
  Stream<String> promptWithContext(List<LlamaMessage> messages, ContextManager contextManager) async* {
    if (modelType != LlamaModelType.chat) {
      throw Exception('This model is not configured for chat. Please use a chat model.');
    }

    // Get the last user message
    final lastUserMessage = messages.lastWhere((msg) => msg.role == 'user');
    
    // Get relevant context
    final context = await contextManager.getRelevantContext(lastUserMessage.content);
    
    // Add context to the system message or create one if it doesn't exist
    final hasSystemMessage = messages.any((msg) => msg.role == 'system');
    if (hasSystemMessage) {
      final systemIndex = messages.indexWhere((msg) => msg.role == 'system');
      messages[systemIndex] = LlamaMessage.withRole(
        role: 'system',
        content: '${messages[systemIndex].content}\n\nRelevant context:\n$context',
      );
    } else {
      messages.insert(0, LlamaMessage.withRole(
        role: 'system',
        content: 'Use this context to help answer the question:\n$context',
      ));
    }

    // Use the regular prompt method with the enhanced messages
    await for (final response in prompt(messages)) {
      yield response;
    }
  }
}
