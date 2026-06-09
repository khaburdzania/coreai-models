// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import FoundationModels
import Synchronization
import Tokenizers

/// FoundationModels Adoption for Core AI inference engines.
///
/// Wraps any `InferenceEngine` (pipelined, sequential, or static-shape) and exposes it
/// through the FoundationModels `LanguageModel` protocol. It uses the modern `tokenSequence()`
/// API for efficient streaming token generation.
///
/// ## Engine Selection
/// The engine type is determined by `EngineFactory` based on model structure:
/// - **Pipelined**: GPU-accelerated with double buffering (fastest for GPU models)
/// - **Sequential**: CPU-based synchronous execution (fallback)
/// - **Static-shape**: Neural Engine optimized for chunked static models
///
/// ## Usage
/// ```swift
/// let engine = try await EngineFactory.createEngine(...)
/// let model = CoreAILanguageModel(engine: engine, tokenizer: tokenizer)
/// let session = LanguageModelSession(model: model)
/// ```
public struct CoreAILanguageModel: LanguageModel {
    // MARK: - Properties

    private let engine: any InferenceEngine
    private let tokenizer: any Tokenizer
    private let modelIdentifier: String
    private let samplingConfig: SamplingConfiguration
    private let vocabSize: Int?

    // MARK: - Protocol Requirements

    public typealias Executor = CoreAIExecutor

    public var capabilities: LanguageModelCapabilities {
        if engine.supportsLogits {
            return LanguageModelCapabilities(capabilities: [.toolCalling, .reasoning, .guidedGeneration])
        }
        return LanguageModelCapabilities(capabilities: [.toolCalling, .reasoning])
    }

    public var executorConfiguration: CoreAIExecutor.Configuration {
        CoreAIExecutor.Configuration(
            engine: engine,
            tokenizer: tokenizer,
            modelIdentifier: modelIdentifier,
            samplingConfig: samplingConfig,
            vocabSize: vocabSize
        )
    }

    // MARK: - Initialization

    /// Creates a CoreAILanguageModel by loading a model bundle from the given URL.
    ///
    /// This convenience initializer handles the full pipeline: asset loading, engine creation
    /// (auto-detected based on model structure), and tokenizer initialization.
    ///
    /// ```swift
    /// let model = try await CoreAILanguageModel(resourcesAt: url)
    /// let session = LanguageModelSession(model: model)
    /// ```
    ///
    /// - Parameter url: URL to the model bundle directory.
    /// - Parameter variant: Engine variant override (e.g. "coreai-sequential",
    ///   "ane"). Nil for auto-detect from model structure.
    /// - Parameter kvCacheStrategy: KV cache memory strategy. Defaults to
    ///   `.auto` (256-token initial size for dynamic models). Pass
    ///   `.fixedSize` to pre-allocate at full `maxContextLength`.
    /// - Throws: If the asset bundle is invalid, engine creation fails, or tokenizer loading fails.
    public init(
        resourcesAt url: URL,
        variant: String? = nil,
        kvCacheStrategy: KVCacheStrategy = .auto
    ) async throws {
        let runner = try CoreAIRunner(
            contentsOf: url,
            variant: variant,
            kvCacheStrategy: kvCacheStrategy
        )
        self = try await runner.makeLanguageModel()
    }

    init(
        engine: any InferenceEngine,
        tokenizer: any Tokenizer,
        modelIdentifier: String = "coreai-model",
        samplingConfig: SamplingConfiguration = .greedy,
        vocabSize: Int? = nil
    ) {
        self.engine = engine
        self.tokenizer = tokenizer
        self.modelIdentifier = modelIdentifier
        self.samplingConfig = samplingConfig
        self.vocabSize = vocabSize
    }

    // MARK: - Executor

    public struct CoreAIExecutor: LanguageModelExecutor {
        public typealias Model = CoreAILanguageModel

        public struct Configuration: Hashable, Sendable {
            fileprivate let engine: any InferenceEngine
            fileprivate let tokenizer: any Tokenizer
            fileprivate let modelIdentifier: String
            fileprivate let samplingConfig: SamplingConfiguration
            fileprivate let vocabSize: Int?

            public static func == (lhs: Configuration, rhs: Configuration) -> Bool {
                lhs.modelIdentifier == rhs.modelIdentifier
                    && lhs.samplingConfig == rhs.samplingConfig
            }

            public func hash(into hasher: inout Hasher) {
                hasher.combine(modelIdentifier)
                hasher.combine(samplingConfig)
            }
        }

        // MARK: - Properties

        private let engine: any InferenceEngine
        private let tokenizer: any Tokenizer
        private let modelIdentifier: String
        private let samplingConfig: SamplingConfiguration
        private let vocabSize: Int?
        /// Open / close marker pair the model uses for chain-of-thought
        /// blocks, discovered from the tokenizer's known token ids at init
        /// (see `detectThinkingMarkers`). For models that don't emit
        /// reasoning, the markers still default to `<think>`/`</think>` and
        /// the parser passes everything through as `.text`.
        private let thinkingMarkers: (open: String, close: String)
        /// Open / close marker pair the model uses for tool call blocks,
        /// discovered from the tokenizer's known token ids at init
        /// (see `detectToolCallMarkers`). nil when the model's tokenizer
        /// has no tool call tokens.
        private let toolCallMarkers: (open: String, close: String)?

        // MARK: - Initialization

        public init(configuration: Configuration) throws {
            self.engine = configuration.engine
            self.tokenizer = configuration.tokenizer
            self.modelIdentifier = configuration.modelIdentifier
            self.samplingConfig = configuration.samplingConfig
            self.vocabSize = configuration.vocabSize
            self.thinkingMarkers = Self.detectThinkingMarkers(configuration.tokenizer)
            self.toolCallMarkers = Self.detectToolCallMarkers(configuration.tokenizer)
        }

        /// Probes the tokenizer for known reasoning marker pairs. Each
        /// candidate pair is verified to exist as added/special tokens via
        /// `convertTokenToId(_:)` — only models that actually have these
        /// tokens in their vocab match. First match wins; falls back to
        /// `<think>`/`</think>` so the parser is harmless on models that
        /// don't emit reasoning markup at all.
        ///
        /// Add a new pair here when onboarding a model with different
        /// markers. For models with non-pair-symmetric formats (e.g.
        /// gpt-oss / Harmony), a different parser is needed; this one
        /// covers the `<open>...</close>` shape.
        private static func detectThinkingMarkers(
            _ tokenizer: any Tokenizer
        ) -> (open: String, close: String) {
            let candidates: [(open: String, close: String)] = [
                ("<think>", "</think>"),
                ("<|reasoning_start|>", "<|reasoning_end|>"),
            ]
            for pair in candidates {
                if tokenizer.convertTokenToId(pair.open) != nil,
                    tokenizer.convertTokenToId(pair.close) != nil
                {
                    return pair
                }
            }
            return ("<think>", "</think>")
        }

        /// Probes the tokenizer for known tool call marker pairs. Each
        /// candidate tag-pair is verified to exist as special tokens via
        /// `convertTokenToId(_:)`. Returns nil when the model's tokenizer
        /// has no tool call tokens at all.
        private static func detectToolCallMarkers(
            _ tokenizer: any Tokenizer
        ) -> (open: String, close: String)? {
            // Standard tag-pair formats — both markers must be special tokens.
            let tagPairs: [(open: String, close: String)] = [
                ("<tool_call>", "</tool_call>"),
                ("<function_calls>", "</function_calls>"),
            ]
            for pair in tagPairs where tokenizer.convertTokenToId(pair.open) != nil
                && tokenizer.convertTokenToId(pair.close) != nil
            {
                return pair
            }
            // Mistral: [TOOL_CALLS] is a special token but has no paired close token.
            if tokenizer.convertTokenToId("[TOOL_CALLS]") != nil {
                return (open: "[TOOL_CALLS]", close: "\n")
            }
            return nil
        }

        // MARK: - Prewarm

        public func prewarm(transcript: Transcript) throws {
            // Use engine's warmup method - blocks until warmup completes.
            //
            // We dispatch async work onto a dedicated DispatchQueue instead of using
            // Task { } + semaphore.wait(). This avoids deadlock: if prewarm() is called
            // from a Swift Concurrency cooperative thread, semaphore.wait() would block
            // a cooperative thread while Task { } needs one to run — thread starvation.
            //
            // With DispatchQueue, the async work runs on a GCD thread (not the cooperative
            // pool), so semaphore.wait() on the calling thread is safe.
            let semaphore = DispatchSemaphore(value: 0)
            let warmupError: Mutex<(any Error)?> = Mutex(nil)

            let queue = DispatchQueue(label: "com.coreai.prewarm")
            queue.async {
                Task {
                    do {
                        try await self.engine.warmup(queryLength: 1, sampling: nil)
                    } catch {
                        warmupError.withLock({ $0 = error })
                    }
                    semaphore.signal()
                }
            }

            semaphore.wait()

            if let error: any Error = warmupError.withLock(\.self) {
                throw error
            }
        }

        // MARK: - respond(to:model:streamingInto:) — new channel-based API

        public nonisolated(nonsending) func respond(
            to request: LanguageModelExecutorGenerationRequest,
            model: CoreAILanguageModel,
            streamingInto channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            // Tokenization span
            let tokenizationSpan = InstrumentsProfiler.beginTokenization(inputLength: 0)
            guard
                let promptTokens = Self.transcriptToTokens(
                    Array(request.transcript),
                    using: tokenizer,
                    tools: request.enabledToolDefinitions,
                    component: "CoreAIExecutor"
                )
            else {
                tokenizationSpan.end()
                throw LanguageModelError.unsupportedTranscriptContent(
                    .init(
                        unsupportedContent: Array(request.transcript),
                        debugDescription: "CoreAI could not tokenize the conversation transcript."
                    )
                )
            }
            tokenizationSpan.end()

            CLILogger.log("Tokenized \(promptTokens.count) tokens", component: "CoreAIExecutor")

            let effectiveSamplingConfig = createSamplingConfig(from: request.generationOptions)
            let maxTokens = request.generationOptions.maximumResponseTokens ?? 512

            // Reset engine state for new generation
            try await engine.reset()

            // FoundationModels now threads entry identity itself based on event
            // ordering — we no longer mint an entryID and pass it down.

            // Check if guided generation is requested
            if let schema = request.schema {
                try await respondConstrained(
                    schema: schema,
                    promptTokens: promptTokens,
                    samplingConfig: effectiveSamplingConfig,
                    maxTokens: maxTokens,
                    channel: channel
                )
            } else {
                try await respondVanilla(
                    promptTokens: promptTokens,
                    samplingConfig: effectiveSamplingConfig,
                    maxTokens: maxTokens,
                    channel: channel
                )
            }
        }

        // MARK: - Vanilla Generation (no schema)

        private func respondVanilla(
            promptTokens: [Int],
            samplingConfig: SamplingConfiguration,
            maxTokens: Int,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            let tokenStream = try engine.generate(
                with: promptTokens.map(Int32.init),
                samplingConfiguration: samplingConfig,
                inferenceOptions: InferenceOptions(maxTokens: maxTokens)
            )

            let eosTokenId = tokenizer.eosTokenId
            // Incremental-decode buffer. After a clean emit, one token is
            // retained as context for the next step (see below). During a
            // multi-byte sequence that hasn't decoded cleanly yet, multiple
            // tokens accumulate until the sequence is complete. In the steady
            // state the buffer holds at most 2 tokens, so tokenizer.decode
            // is O(1) per step.
            var pendingTokens: [Int32] = []
            var previousDecodedText: String = ""
            var tokenStep: Int = 0
            // Segments the decoded stream into `.text` and `.reasoning`
            // events on the fly. Reasoning content (model's chain-of-thought
            // emitted inside the configured open/close markers) is routed
            // to a top-level `.reasoning(...)` channel event so it lands as
            // its own `Transcript.Reasoning` entry, not mixed into the
            // user-facing `Transcript.Response`. Markers were resolved at
            // executor init from the tokenizer's known token ids.
            var thinkParser = ThinkTagParser(
                open: thinkingMarkers.open,
                close: thinkingMarkers.close
            )
            // Routes tool call markup to .toolCalls(...) channel events.
            // nil when the model's tokenizer has no tool call tokens.
            var toolCallParser: ToolCallParser? = toolCallMarkers.map {
                ToolCallParser(open: $0.open, close: $0.close)
            }
            var generatedTokenCount: Int = 0

            for try await output in tokenStream {
                let token = output.tokenId
                if let eos = eosTokenId, Int(token) == eos { break }

                pendingTokens.append(token)
                tokenStep += 1
                generatedTokenCount += 1

                let decodeSpan = InstrumentsProfiler.beginDecode(step: tokenStep)
                let decodedText = tokenizer.decode(tokens: pendingTokens.map { Int($0) })
                decodeSpan.end()

                let common = decodedText.commonPrefix(with: previousDecodedText)
                let delta = String(decodedText.dropFirst(common.count))
                // Check for replacement char on the full `decodedText`, not on
                // `delta`. Some tokenizers emit one U+FFFD per attempted decode
                // of an incomplete multi-byte sequence (rather than one per
                // bad byte), so two consecutive partial tokens can produce
                // identical "\u{FFFD}" strings — making `delta` empty and
                // hiding the still-incomplete state. Checking `decodedText`
                // catches that case.
                let hasReplacementChar = decodedText.unicodeScalars.contains { $0 == "\u{FFFD}" }

                if hasReplacementChar {
                    // UTF-8 bytes don't form a clean character yet. Hold the
                    // token and wait for the next iteration to extend the
                    // buffer; don't drop or advance.
                    await channel.send(
                        .response(action: .appendText("", tokenCount: 1))
                    )
                    previousDecodedText = decodedText
                    continue
                }

                for event in thinkParser.consume(delta) {
                    await dispatch(event, toolCallParser: &toolCallParser, channel: channel)
                }

                // Retain the last token as O(1) context for the next decode.
                // SentencePiece needs at least one prior token to infer the leading
                // ▁ (space) on the following token; clearing to empty decodes each
                // new token in isolation and drops inter-word spaces.
                // Keeping one token bounds re-decode cost to 2 tokens per step.
                // Safe for all supported tokenizers: decode([last]) is a prefix of
                // decode([last, next]) when addPrefixSpace=true (Mistral, Llama, Qwen)
                // and for ByteLevel tokenizers (GPT-2 style) where spaces are direct bytes.
                if let last = pendingTokens.last {
                    pendingTokens = [last]
                    previousDecodedText = tokenizer.decode(tokens: [Int(last)])
                } else {
                    pendingTokens.removeAll(keepingCapacity: true)
                    previousDecodedText = ""
                }
            }

            // Flush parsers — drains any content held back waiting for a marker.
            for event in thinkParser.flush() {
                await dispatch(event, toolCallParser: &toolCallParser, channel: channel)
            }
            if var tcp = toolCallParser {
                for event in tcp.flush() {
                    await dispatchToolCallEvent(event, channel: channel)
                }
                toolCallParser = tcp
            }

            // Usage telemetry placeholder — awaiting Usage(input:output:) API.
            _ = promptTokens.count
            _ = generatedTokenCount

            // Yield to let the engine's tokenSequence Task finish cleanup
            // (putBackEngine, state reset, etc.) before the next respond().
            await Task.yield()
        }

        // MARK: - Event Dispatch

        /// Routes a parser event to the matching FoundationModels channel event.
        /// Text is forwarded to the tool call parser (if present) or emitted as
        /// `.response(...).appendText`. Reasoning becomes a top-level
        /// `.reasoning(...).appendText`. Reasoning is a sibling of
        /// response/tool-calls in the new API (not nested under response)
        /// because at parse time we don't yet know whether the model will
        /// follow the thought block with a response or a tool call.
        ///
        /// We deliberately do not pass `entryID` — FoundationModels threads
        /// entry identity itself based on event ordering.
        private func dispatch(
            _ event: ThinkTagParser.Event,
            toolCallParser: inout ToolCallParser?,
            channel: LanguageModelExecutorGenerationChannel
        ) async {
            switch event {
            case .reasoning(let text):
                await channel.send(
                    .reasoning(action: .appendText(text, tokenCount: 1))
                )
            case .text(let text):
                if var tcp = toolCallParser {
                    for toolEvent in tcp.consume(text) {
                        await dispatchToolCallEvent(toolEvent, channel: channel)
                    }
                    toolCallParser = tcp
                } else if !text.isEmpty {
                    await channel.send(
                        .response(action: .appendText(text, tokenCount: 1))
                    )
                }
            }
        }

        private func dispatchToolCallEvent(
            _ event: ToolCallParser.Event,
            channel: LanguageModelExecutorGenerationChannel
        ) async {
            switch event {
            case .text(let text):
                if !text.isEmpty {
                    await channel.send(
                        .response(action: .appendText(text, tokenCount: 1))
                    )
                }
            case .toolCall(let id, let name, let argsJSON):
                let tokenCount = max(1, argsJSON.utf8.count / 4)
                CLILogger.log(
                    "ToolCallParser: dispatching tool call id=\(id) name=\(name) args=\(argsJSON)",
                    component: "CoreAIExecutor")
                await channel.send(
                    .toolCalls(
                        action: .toolCall(
                            id: id,
                            name: name,
                            action: .appendArguments(argsJSON, tokenCount: tokenCount)
                        )
                    )
                )
            }
        }

        // MARK: - Constrained Generation (with schema)

        private func respondConstrained(
            schema: GenerationSchema,
            promptTokens: [Int],
            samplingConfig: SamplingConfiguration,
            maxTokens: Int,
            channel: LanguageModelExecutorGenerationChannel
        ) async throws {
            let schemaData = try JSONEncoder().encode(schema)

            guard let jsonSchema = String(data: schemaData, encoding: .utf8) else {
                preconditionFailure("GenerationSchema JSON encoding produced invalid UTF-8")
            }

            let strategy = ConstrainedDecodingStrategy(jsonSchema: jsonSchema, vocabSize: vocabSize)
            let stopSequences = StopSequences(for: tokenizer)

            let stream = strategy.decode(
                from: .tokens(promptTokens),
                tokenizer: tokenizer,
                inferenceEngine: engine,
                samplingConfiguration: samplingConfig,
                options: InferenceOptions(maxTokens: maxTokens),
                stopSequences: stopSequences
            )

            // Bridge AsyncThrowingStream -> LanguageModelExecutorGenerationChannel
            var generatedTokenCount = 0
            for try await result in stream {
                generatedTokenCount += 1
                await channel.send(
                    .response(action: .appendText(result.text, tokenCount: 1))
                )
            }

            // Usage telemetry placeholder — awaiting Usage(input:output:) API.
            _ = promptTokens.count
            _ = generatedTokenCount

            // Yield to let the engine's tokenSequence Task finish cleanup
            // (putBackEngine, state reset, etc.) before the next respond().
            await Task.yield()
        }

        // MARK: - Transcript → Tokens

        /// Converts transcript entries to tokens using the provided tokenizer.
        ///
        /// Handles all entry types including prior tool calls and tool outputs.
        /// Tool definitions are forwarded to `applyChatTemplate` so the model
        /// sees the available functions in the system prompt.
        static func transcriptToTokens(
            _ entries: [Transcript.Entry],
            using tokenizer: any Tokenizer,
            tools: [Transcript.ToolDefinition] = [],
            component: String = "CoreAIExecutor"
        ) -> [Int]? {
            var messages: [Message] = []

            for entry in entries {
                switch entry {
                case .instructions(let instructions):
                    let text = instructions.segments.compactMap {
                        if case .text(let t) = $0 { return t.content }
                        return nil
                    }.joined(separator: "\n")
                    if !text.isEmpty {
                        messages.append(["role": "system", "content": text])
                    }

                case .prompt(let prompt):
                    let text = prompt.segments.compactMap {
                        if case .text(let t) = $0 { return t.content }
                        return nil
                    }.joined()
                    if !text.isEmpty {
                        messages.append(["role": "user", "content": text])
                    }

                case .response(let response):
                    let text = response.segments.compactMap {
                        if case .text(let t) = $0 { return t.content }
                        return nil
                    }.joined()
                    if !text.isEmpty {
                        messages.append(["role": "assistant", "content": text])
                    }

                case .toolCalls(let toolCalls):
                    // Assistant turn that invoked tools — map to OpenAI-style tool_calls array.
                    var calls: [[String: any Sendable]] = []
                    for call in toolCalls {
                        let function: [String: any Sendable] = [
                            "name": call.toolName,
                            "arguments": call.arguments.jsonString,
                        ]
                        calls.append([
                            "id": call.id,
                            "type": "function",
                            "function": function,
                        ])
                    }
                    // Tool-calling assistant turns have no text body.
                    messages.append([
                        "role": "assistant",
                        "content": "" as any Sendable,
                        "tool_calls": calls as any Sendable,
                    ])

                case .toolOutput(let output):
                    // Tool result turn — map to OpenAI-style tool role message.
                    let content = output.segments.compactMap { segment -> String? in
                        if case .text(let t) = segment { return t.content }
                        return nil
                    }.joined()
                    messages.append([
                        "role": "tool",
                        "tool_call_id": output.id,
                        "name": output.toolName,
                        "content": content,
                    ])

                case .reasoning:
                    // Don't echo the model's prior reasoning back into the prompt.
                    continue

                @unknown default:
                    continue
                }
            }

            if messages.isEmpty { return nil }

            let toolSpecs: [ToolSpec]? = tools.isEmpty ? nil : tools.compactMap { makeToolSpec(from: $0) }

            do {
                CLILogger.log("Applying chat template via tokenizer", component: component)
                return try tokenizer.applyChatTemplate(messages: messages, tools: toolSpecs)
            } catch {
                CLILogger.log(
                    "Failed to apply chat template: \(error), falling back to simple encoding",
                    component: component)
                let text = messages.compactMap { $0["content"] as? String }.joined(separator: "\n")
                return tokenizer.encode(text: text)
            }
        }

        /// Converts a `ToolDefinition` into the `ToolSpec` format expected by
        /// `applyChatTemplate`. The `parameters` `GenerationSchema` is encoded to
        /// JSON then recursively converted to `[String: any Sendable]` so the
        /// Jinja template engine can walk the nested structure.
        private static func makeToolSpec(from definition: Transcript.ToolDefinition) -> ToolSpec? {
            guard
                let schemaData = try? JSONEncoder().encode(definition.parameters),
                let rawObj = try? JSONSerialization.jsonObject(with: schemaData),
                let paramsAny = rawObj as? [String: Any]
            else {
                CLILogger.log(
                    "Failed to encode parameters for tool '\(definition.name)'",
                    component: "CoreAIExecutor")
                return nil
            }
            let function: [String: any Sendable] = [
                "name": definition.name,
                "description": definition.description,
                "parameters": convertToSendable(paramsAny),
            ]
            return ["type": "function", "function": function]
        }

        /// Recursively converts a JSON-deserialized `Any` tree to `any Sendable`.
        ///
        /// `JSONSerialization` returns NS-bridged types (`NSDictionary`, `NSArray`,
        /// `NSNumber`) that aren't typed as `Sendable`. This converts them to
        /// pure-Swift equivalents so they can be placed in `[String: any Sendable]`
        /// without compiler warnings and the Jinja `Value(any:)` handler processes
        /// them correctly.
        private static func convertToSendable(_ value: Any) -> any Sendable {
            switch value {
            case let dict as [String: Any]:
                return dict.reduce(into: [String: any Sendable]()) { result, pair in
                    result[pair.key] = convertToSendable(pair.value)
                }
            case let array as [Any]:
                return array.map { convertToSendable($0) }
            case let str as String:
                return str
            case let num as NSNumber:
                if CFGetTypeID(num) == CFBooleanGetTypeID() { return num.boolValue }
                let d = num.doubleValue
                if d == d.rounded() && !d.isInfinite { return num.intValue }
                return d
            default:
                return String(describing: value)
            }
        }

        // MARK: - Helper Methods

        private func createSamplingConfig(from options: GenerationOptions) -> SamplingConfiguration {
            if let temperature = options.temperature {
                return SamplingConfiguration(temperature: temperature)
            }
            return samplingConfig
        }
    }
}
