// Copyright © 2026 Apple, Inc. All rights reserved.

import Testing

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

/// Coverage for `ToolCallParser`. Pins the behaviors most likely to regress:
/// passthrough on non-tool models, complete block in one chunk, marker
/// straddling two consumes, multiple sequential calls, mixed content, malformed
/// JSON being silently dropped, and the EOS flush path.
@Suite("ToolCallParser")
struct ToolCallParserTests {
    @Test("No markers — all input emitted as .text")
    func passthroughWhenNoMarkers() {
        var parser = ToolCallParser()
        let events = parser.consume("Hello, world!") + parser.flush()
        #expect(texts(events) == ["Hello, world!"])
        #expect(toolCalls(events).isEmpty)
    }

    @Test("Complete tool call block in one consume")
    func fullBlockInOneConsume() {
        var parser = ToolCallParser()
        let json = #"{"name":"get_weather","arguments":{"city":"London"}}"#
        let events = parser.consume("<tool_call>\(json)</tool_call>") + parser.flush()
        #expect(texts(events).isEmpty)
        let calls = toolCalls(events)
        #expect(calls.count == 1)
        #expect(calls[0].name == "get_weather")
        #expect(calls[0].argsJSON.contains("London"))
        #expect(!calls[0].id.isEmpty)
    }

    @Test("Text before and after a tool call")
    func textAroundToolCall() {
        var parser = ToolCallParser()
        let json = #"{"name":"add","arguments":{"a":1,"b":2}}"#
        let events = parser.consume("Prefix<tool_call>\(json)</tool_call>Suffix") + parser.flush()
        #expect(texts(events) == ["Prefix", "Suffix"])
        let calls = toolCalls(events)
        #expect(calls.count == 1)
        #expect(calls[0].name == "add")
    }

    @Test("Open marker straddling two consumes — buffer holds partial match")
    func openMarkerStraddlesTwoConsumes() {
        var parser = ToolCallParser()
        let json = #"{"name":"ping","arguments":{}}"#
        // First chunk ends in "<tool_" — a prefix of the open marker.
        var events = parser.consume("before<tool_")
        // No tool call text should leak yet.
        #expect(texts(events) == ["before"])
        events += parser.consume("call>\(json)</tool_call>")
        events += parser.flush()
        #expect(texts(events) == ["before"])
        let calls = toolCalls(events)
        #expect(calls.count == 1)
        #expect(calls[0].name == "ping")
    }

    @Test("Close marker straddling two consumes")
    func closeMarkerStraddlesTwoConsumes() {
        var parser = ToolCallParser()
        let jsonStart = #"{"name":"search","arguments":{"q":"swift"}}"#
        // Feed open marker + JSON + partial close.
        var events = parser.consume("<tool_call>\(jsonStart)</tool_")
        #expect(toolCalls(events).isEmpty)
        events += parser.consume("call>")
        events += parser.flush()
        let calls = toolCalls(events)
        #expect(calls.count == 1)
        #expect(calls[0].name == "search")
    }

    @Test("Multiple sequential tool calls")
    func multipleSequentialToolCalls() {
        var parser = ToolCallParser()
        let json1 = #"{"name":"first","arguments":{"x":1}}"#
        let json2 = #"{"name":"second","arguments":{"y":2}}"#
        let input = "<tool_call>\(json1)</tool_call><tool_call>\(json2)</tool_call>"
        let events = parser.consume(input) + parser.flush()
        let calls = toolCalls(events)
        #expect(calls.count == 2)
        #expect(calls[0].name == "first")
        #expect(calls[1].name == "second")
    }

    @Test("Each tool call gets a unique ID")
    func uniqueIDs() {
        var parser = ToolCallParser()
        let json = #"{"name":"fn","arguments":{}}"#
        let input = "<tool_call>\(json)</tool_call><tool_call>\(json)</tool_call>"
        let calls = toolCalls(parser.consume(input) + parser.flush())
        #expect(calls.count == 2)
        #expect(calls[0].id != calls[1].id)
    }

    @Test("Arguments as a string (model stringifies JSON object)")
    func argumentsAsString() {
        var parser = ToolCallParser()
        // Some models emit arguments as a JSON-encoded string instead of an object.
        let json = #"{"name":"fn","arguments":"{\"key\":\"val\"}"}"#
        let events = parser.consume("<tool_call>\(json)</tool_call>") + parser.flush()
        let calls = toolCalls(events)
        #expect(calls.count == 1)
        #expect(calls[0].argsJSON.contains("val"))
    }

    @Test("Malformed JSON — event silently dropped")
    func malformedJSONDropped() {
        var parser = ToolCallParser()
        let events = parser.consume("<tool_call>not json</tool_call>after") + parser.flush()
        #expect(toolCalls(events).isEmpty)
        // Text after the bad block still passes through.
        #expect(texts(events) == ["after"])
    }

    @Test("Missing 'name' field — event silently dropped")
    func missingNameDropped() {
        var parser = ToolCallParser()
        let json = #"{"arguments":{"x":1}}"#
        let events = parser.consume("<tool_call>\(json)</tool_call>after") + parser.flush()
        #expect(toolCalls(events).isEmpty)
        #expect(texts(events) == ["after"])
    }

    @Test("Unclosed tool_call at EOS — silently dropped, not emitted as text")
    func unclosedBlockAtEndOfStreamDropped() {
        var parser = ToolCallParser()
        let events = parser.consume("<tool_call>{\"name\":\"fn\",\"arguments\":{}}") + parser.flush()
        #expect(toolCalls(events).isEmpty)
        #expect(texts(events).isEmpty)
    }

    @Test("Mistral array format: [TOOL_CALLS] [{...}] parsed via flush at EOS")
    func mistralArrayFormatViaEOSFlush() {
        // Mistral uses "\n" as the synthetic close; the array arrives before EOS with no trailing newline.
        var parser = ToolCallParser(open: "[TOOL_CALLS]", close: "\n")
        let json = #"[{"name":"get_weather","arguments":{"city":"London"}}]"#
        let events = parser.consume("[TOOL_CALLS] \(json)") + parser.flush()
        let calls = toolCalls(events)
        #expect(calls.count == 1)
        #expect(calls[0].name == "get_weather")
        #expect(calls[0].argsJSON.contains("London"))
    }

    @Test("Mistral array format: trailing newline triggers close")
    func mistralArrayFormatWithNewline() {
        var parser = ToolCallParser(open: "[TOOL_CALLS]", close: "\n")
        let json = #"[{"name":"search","arguments":{"q":"swift"}}]"#
        let events = parser.consume("[TOOL_CALLS] \(json)\n") + parser.flush()
        let calls = toolCalls(events)
        #expect(calls.count == 1)
        #expect(calls[0].name == "search")
    }

    @Test("Mistral array format: multiple tool calls in one array")
    func mistralArrayFormatMultipleCalls() {
        var parser = ToolCallParser(open: "[TOOL_CALLS]", close: "\n")
        let json = #"[{"name":"first","arguments":{"x":1}},{"name":"second","arguments":{"y":2}}]"#
        let events = parser.consume("[TOOL_CALLS] \(json)\n") + parser.flush()
        let calls = toolCalls(events)
        #expect(calls.count == 2)
        #expect(calls[0].name == "first")
        #expect(calls[1].name == "second")
    }

    @Test("Whitespace-trimmed JSON body is still parsed")
    func whitespaceAroundJSON() {
        var parser = ToolCallParser()
        let json = "  \n{\"name\":\"trim_test\",\"arguments\":{}}\n  "
        let events = parser.consume("<tool_call>\(json)</tool_call>") + parser.flush()
        let calls = toolCalls(events)
        #expect(calls.count == 1)
        #expect(calls[0].name == "trim_test")
    }

    @Test("Custom open/close markers")
    func customMarkers() {
        var parser = ToolCallParser(open: "<function_calls>", close: "</function_calls>")
        let json = #"{"name":"custom","arguments":{"v":42}}"#
        let events = parser.consume("<function_calls>\(json)</function_calls>") + parser.flush()
        let calls = toolCalls(events)
        #expect(calls.count == 1)
        #expect(calls[0].name == "custom")
    }

    // MARK: - Helpers

    private func texts(_ events: [ToolCallParser.Event]) -> [String] {
        events.compactMap { if case .text(let s) = $0 { return s } else { return nil } }
    }

    private struct ToolCallInfo { let id: String; let name: String; let argsJSON: String }

    private func toolCalls(_ events: [ToolCallParser.Event]) -> [ToolCallInfo] {
        events.compactMap {
            if case .toolCall(let id, let name, let argsJSON) = $0 {
                return ToolCallInfo(id: id, name: name, argsJSON: argsJSON)
            }
            return nil
        }
    }
}

#endif
