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
            return LanguageModelCapabilities(capabilities: [.guidedGeneration])
        }
        return LanguageModelCapabilities(capabilities: [])
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

    // MARK: - Helper Methods

    /// Converts transcript entries to tokens using the provided tokenizer.
    /// Shared implementation used by both `CoreAILanguageModel` and `CoreAIExecutor`.
    static func transcriptToTokens(
        _ entries: [Transcript.Entry],
        using tokenizer: any Tokenizer,
        component: String = "CoreAILanguageModel"
    ) -> [Int]? {
        var messages: [[String: String]] = []

        for entry in entries {
            switch entry {
            case .instructions(let instructions):
                for segment in instructions.segments {
                    if case .text(let text) = segment {
                        messages.append(["role": "system", "content": text.content])
                    }
                }

            case .prompt(let prompt):
                for segment in prompt.segments {
                    if case .text(let text) = segment {
                        messages.append(["role": "user", "content": text.content])
                    }
                }

            case .response(let response):
                for segment in response.segments {
                    if case .text(let text) = segment {
                        messages.append(["role": "assistant", "content": text.content])
                    }
                }

            default:
                continue
            }
        }

        if !messages.isEmpty {
            do {
                CLILogger.log("Applying chat template via tokenizer", component: component)
                return try tokenizer.applyChatTemplate(messages: messages)
            } catch {
                CLILogger.log(
                    "Failed to apply chat template: \(error), falling back to simple encoding",
                    component: component)
                let text = messages.compactMap { $0["content"] }.joined(separator: "\n")
                return tokenizer.encode(text: text)
            }
        }

        return nil
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

        // MARK: - Initialization (new API)

        public init(configuration: Configuration) throws {
            self.engine = configuration.engine
            self.tokenizer = configuration.tokenizer
            self.modelIdentifier = configuration.modelIdentifier
            self.samplingConfig = configuration.samplingConfig
            self.vocabSize = configuration.vocabSize
            self.thinkingMarkers = Self.detectThinkingMarkers(configuration.tokenizer)
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
                let promptTokens = CoreAILanguageModel.transcriptToTokens(
                    Array(request.transcript),
                    using: tokenizer,
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
            // ordering — we no longer mint an entryID and pass it down. Same for
            // metadata: updateMetadata is available on every entry type, but
            // we don't emit any from here today (metadata flows from the
            // upstream PromptCompletion pipeline once that lands).

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
            var parser = ThinkTagParser(
                open: thinkingMarkers.open,
                close: thinkingMarkers.close
            )
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

                for event in parser.consume(delta) {
                    await dispatch(event, channel: channel)
                }

                // Retain the last token as O(1) context for the next decode.
                // SentencePiece needs at least one prior token to infer the leading
                // ▁ (space) on the following token; clearing to empty decodes each
                // new token in isolation and drops inter-word spaces.
                // Keeping one token bounds re-decode cost to 2 tokens per step.
                // Safe for all supported tokenizers: decode([last]) is a prefix of
                // decode([last, next]) when addPrefixSpace=true (Mistral, Qwen)
                // and for ByteLevel tokenizers (GPT-2 style) where spaces are direct bytes.
                if let last = pendingTokens.last {
                    pendingTokens = [last]
                    previousDecodedText = tokenizer.decode(tokens: [Int(last)])
                } else {
                    pendingTokens.removeAll(keepingCapacity: true)
                    previousDecodedText = ""
                }
            }

            // Flush any buffered content the parser was holding back for a
            // possible marker match. Without this, content right at the EOS
            // boundary (or inside an unclosed `<think>` block) would be lost.
            for event in parser.flush() {
                await dispatch(event, channel: channel)
            }

            // Usage telemetry placeholder — awaiting Usage(input:output:) API.
            _ = promptTokens.count
            _ = generatedTokenCount

            // Yield to let the engine's tokenSequence Task finish cleanup
            // (putBackEngine, state reset, etc.) before the next respond().
            await Task.yield()
        }

        /// Routes a parser event to the matching FoundationModels channel
        /// event. Text becomes `.response(...).appendText`; reasoning becomes
        /// a top-level `.reasoning(...).appendText`. Reasoning is a sibling
        /// of response/tool-calls in the new API (not nested under response)
        /// because at parse time we don't yet know whether the model will
        /// follow the thought block with a response or a tool call.
        ///
        /// We deliberately do not pass `entryID` — FoundationModels threads
        /// entry identity itself based on event ordering.
        private func dispatch(
            _ event: ThinkTagParser.Event,
            channel: LanguageModelExecutorGenerationChannel
        ) async {
            switch event {
            case .text(let text):
                await channel.send(
                    .response(action: .appendText(text, tokenCount: 1))
                )
            case .reasoning(let text):
                await channel.send(
                    .reasoning(action: .appendText(text, tokenCount: 1))
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

            print("‹CoreAI-diag› constrained schema (\(jsonSchema.count) chars): \(jsonSchema.prefix(600))")

            // Bridge AsyncThrowingStream -> LanguageModelExecutorGenerationChannel
            var generatedTokenCount = 0
            var fullText = ""
            for try await result in stream {
                generatedTokenCount += 1
                fullText += result.text
                await channel.send(
                    .response(action: .appendText(result.text, tokenCount: 1))
                )
            }
            print("‹CoreAI-diag› constrained output (\(generatedTokenCount) tokens): >>>\(fullText)<<<")

            // Usage telemetry placeholder — awaiting Usage(input:output:) API.
            _ = promptTokens.count
            _ = generatedTokenCount

            // Yield to let the engine's tokenSequence Task finish cleanup
            // (putBackEngine, state reset, etc.) before the next respond().
            await Task.yield()
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
