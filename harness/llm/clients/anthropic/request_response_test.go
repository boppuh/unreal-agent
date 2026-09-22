package anthropic

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/unreallabsai/unreal-agent/harness/llm"
)

func TestRequestParamsConvertsConversation(t *testing.T) {
	maxTokens := int64(4096)
	request := llm.Request{
		Model: llm.Model{ID: "claude-test", MaxOutputTokens: &maxTokens, ReasoningEffort: llm.ReasoningEffortXHigh},
		Input: []llm.Item{
			{Type: llm.ItemMessage, Data: llm.Message{Role: llm.RoleSystem, Text: "Be exact."}},
			{Type: llm.ItemMessage, Data: llm.Message{Role: llm.RoleUser, Text: "inspect"}},
			{Type: llm.ItemReasoning, Data: llm.Reasoning{Raw: []byte(`{"type":"thinking","thinking":"check","signature":"sig"}`)}},
			{Type: llm.ItemToolCall, Data: llm.ToolCall{CallID: "toolu_1", Name: "Bash", Arguments: `{"command":"pwd"}`}},
			{Type: llm.ItemToolResult, Data: llm.ToolResult{CallID: "toolu_1", Output: []llm.ToolResultOutput{
				{Kind: llm.ToolResultText, Value: "/work"},
				{Kind: llm.ToolResultImage, Value: "data:image/png;base64,aGVsbG8="},
			}}},
		},
		Tools: []llm.Tool{{
			Type: llm.ToolFunction, Name: "Bash", Description: "Run a command",
			Parameters: map[string]any{
				"type": "object", "properties": map[string]any{"command": map[string]any{"type": "string"}}, "required": []any{"command"},
			},
		}},
	}
	params, err := requestParams(request, llm.RequestOptions{CacheKey: "session"})
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(encoded, &body); err != nil {
		t.Fatal(err)
	}
	if body["model"] != "claude-test" || body["max_tokens"] != float64(4096) {
		t.Fatalf("model request = %s", encoded)
	}
	if body["cache_control"].(map[string]any)["type"] != "ephemeral" || body["output_config"].(map[string]any)["effort"] != "max" ||
		body["thinking"].(map[string]any)["type"] != "adaptive" {
		t.Fatalf("cache/effort = %s", encoded)
	}
	system := body["system"].([]any)
	if len(system) != 1 || system[0].(map[string]any)["text"] != "Be exact." {
		t.Fatalf("system = %#v", system)
	}
	messages := body["messages"].([]any)
	if len(messages) != 3 || messages[0].(map[string]any)["role"] != "user" ||
		messages[1].(map[string]any)["role"] != "assistant" || messages[2].(map[string]any)["role"] != "user" {
		t.Fatalf("messages = %#v", messages)
	}
	assistant := messages[1].(map[string]any)["content"].([]any)
	if len(assistant) != 2 || assistant[0].(map[string]any)["signature"] != "sig" || assistant[1].(map[string]any)["type"] != "tool_use" {
		t.Fatalf("assistant content = %#v", assistant)
	}
	result := messages[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	if result["type"] != "tool_result" || len(result["content"].([]any)) != 2 {
		t.Fatalf("tool result = %#v", result)
	}
	tools := body["tools"].([]any)
	if len(tools) != 1 || tools[0].(map[string]any)["description"] != "Run a command" {
		t.Fatalf("tools = %#v", tools)
	}
}

func TestRequestParamsNormalizesRepeatedToolResults(t *testing.T) {
	params, err := requestParams(llm.Request{
		Model: llm.Model{ID: "claude-test"},
		Input: []llm.Item{
			{Type: llm.ItemMessage, Data: llm.Message{Role: llm.RoleUser, Text: "start"}},
			{Type: llm.ItemToolCall, Data: llm.ToolCall{CallID: "toolu_1", Name: "Bash", Arguments: `{}`}},
			{Type: llm.ItemToolResult, Data: llm.ToolResult{CallID: "toolu_1", Output: []llm.ToolResultOutput{{Kind: llm.ToolResultText, Value: "still running"}}}},
			{Type: llm.ItemMessage, Data: llm.Message{Role: llm.RoleAssistant, Text: "I will continue."}},
			{Type: llm.ItemToolResult, Data: llm.ToolResult{CallID: "toolu_1", Output: []llm.ToolResultOutput{{Kind: llm.ToolResultText, Value: "completed"}}}},
		},
	}, llm.RequestOptions{})
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(encoded, &body); err != nil {
		t.Fatal(err)
	}
	messages := body["messages"].([]any)
	if len(messages) != 5 {
		t.Fatalf("messages = %s", encoded)
	}
	firstResult := messages[2].(map[string]any)["content"].([]any)
	if len(firstResult) != 1 || firstResult[0].(map[string]any)["type"] != "tool_result" {
		t.Fatalf("first result = %#v", firstResult)
	}
	followUp := messages[4].(map[string]any)["content"].([]any)
	if len(followUp) != 2 || followUp[0].(map[string]any)["type"] != "text" ||
		followUp[1].(map[string]any)["text"] != "completed" {
		t.Fatalf("follow-up result = %#v", followUp)
	}
}

func TestRequestParamsPlacesCurrentResultBeforeInterveningInput(t *testing.T) {
	params, err := requestParams(llm.Request{
		Model: llm.Model{ID: "claude-test"},
		Input: []llm.Item{
			{Type: llm.ItemMessage, Data: llm.Message{Role: llm.RoleUser, Text: "start"}},
			{Type: llm.ItemToolCall, Data: llm.ToolCall{CallID: "toolu_1", Name: "Bash", Arguments: `{}`}},
			{Type: llm.ItemMessage, Data: llm.Message{Role: llm.RoleUser, Text: "intervening input"}},
			{Type: llm.ItemToolResult, Data: llm.ToolResult{CallID: "toolu_1", Output: []llm.ToolResultOutput{{Kind: llm.ToolResultText, Value: "completed"}}}},
		},
	}, llm.RequestOptions{})
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(encoded, &body); err != nil {
		t.Fatal(err)
	}
	messages := body["messages"].([]any)
	content := messages[len(messages)-1].(map[string]any)["content"].([]any)
	if len(content) != 2 || content[0].(map[string]any)["type"] != "tool_result" ||
		content[1].(map[string]any)["text"] != "intervening input" {
		t.Fatalf("current result was not placed first: %s", encoded)
	}
}

func TestRequestParamsPlacesCurrentResultsBeforeDelayedUpdates(t *testing.T) {
	params, err := requestParams(llm.Request{
		Model: llm.Model{ID: "claude-test"},
		Input: []llm.Item{
			{Type: llm.ItemMessage, Data: llm.Message{Role: llm.RoleUser, Text: "start"}},
			{Type: llm.ItemToolCall, Data: llm.ToolCall{CallID: "toolu_old", Name: "Bash", Arguments: `{}`}},
			{Type: llm.ItemToolResult, Data: llm.ToolResult{CallID: "toolu_old", Output: []llm.ToolResultOutput{{Kind: llm.ToolResultText, Value: "still running"}}}},
			{Type: llm.ItemToolCall, Data: llm.ToolCall{CallID: "toolu_current", Name: "Bash", Arguments: `{}`}},
			{Type: llm.ItemToolResult, Data: llm.ToolResult{CallID: "toolu_old", Output: []llm.ToolResultOutput{{Kind: llm.ToolResultText, Value: "old completed"}}}},
			{Type: llm.ItemToolResult, Data: llm.ToolResult{CallID: "toolu_current", Output: []llm.ToolResultOutput{{Kind: llm.ToolResultText, Value: "current running"}}}},
		},
	}, llm.RequestOptions{})
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(encoded, &body); err != nil {
		t.Fatal(err)
	}
	messages := body["messages"].([]any)
	content := messages[len(messages)-1].(map[string]any)["content"].([]any)
	if len(content) != 3 || content[0].(map[string]any)["type"] != "tool_result" ||
		content[0].(map[string]any)["tool_use_id"] != "toolu_current" ||
		content[1].(map[string]any)["type"] != "text" || content[2].(map[string]any)["text"] != "old completed" {
		t.Fatalf("result/update ordering = %s", encoded)
	}
}

func TestRequestParamsValidatesProviderSpecificValues(t *testing.T) {
	zero := int64(0)
	for _, test := range []struct {
		name    string
		request llm.Request
		match   string
	}{
		{name: "model", request: llm.Request{}, match: "model must be set"},
		{name: "tokens", request: llm.Request{Model: llm.Model{ID: "m", MaxOutputTokens: &zero}}, match: "must be positive"},
		{name: "effort", request: llm.Request{Model: llm.Model{ID: "m", ReasoningEffort: "extreme"}}, match: "unsupported reasoning effort"},
		{name: "hosted tool", request: llm.Request{Model: llm.Model{ID: "m"}, Tools: []llm.Tool{{Type: llm.ToolHosted, Name: "web_search"}}}, match: "unsupported tool type"},
		{name: "foreign reasoning", request: llm.Request{Model: llm.Model{ID: "m"}, Input: []llm.Item{{Type: llm.ItemReasoning, Data: llm.Reasoning{Raw: []byte(`{"type":"reasoning"}`)}}}}, match: "unsupported Anthropic reasoning"},
		{name: "image URL", request: llm.Request{Model: llm.Model{ID: "m"}, Input: []llm.Item{{Type: llm.ItemToolResult, Data: llm.ToolResult{CallID: "c", Output: []llm.ToolResultOutput{{Kind: llm.ToolResultImage, Value: "https://example.com/image.png"}}}}}}, match: "base64 data URL"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := requestParams(test.request, llm.RequestOptions{})
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("error = %v, want %q", err, test.match)
			}
		})
	}
}

func TestResponseConvertsContentStopsAndUsage(t *testing.T) {
	var source anthropicsdk.Message
	if err := json.Unmarshal([]byte(`{
		"id":"msg_1","type":"message","role":"assistant","model":"claude-test",
		"content":[
			{"type":"thinking","thinking":"inspect first","signature":"opaque"},
			{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"pwd"}}
		],
		"stop_reason":"tool_use","stop_sequence":null,
		"usage":{"input_tokens":10,"cache_creation_input_tokens":3,"cache_read_input_tokens":4,"output_tokens":8,"output_tokens_details":{"thinking_tokens":5}}
	}`), &source); err != nil {
		t.Fatal(err)
	}
	converted, err := response(source)
	if err != nil {
		t.Fatal(err)
	}
	if converted.ID != "msg_1" || converted.Stop != llm.StopComplete || len(converted.Output) != 2 {
		t.Fatalf("response = %#v", converted)
	}
	reasoning := converted.Output[0].Data.(llm.Reasoning)
	if !reflect.DeepEqual(reasoning.Summary, []string{"inspect first"}) || !strings.Contains(string(reasoning.Raw), `"signature":"opaque"`) {
		t.Fatalf("reasoning = %#v", reasoning)
	}
	call := converted.Output[1].Data.(llm.ToolCall)
	if call.CallID != "toolu_1" || call.Name != "Bash" || call.Arguments != `{"command":"pwd"}` {
		t.Fatalf("tool call = %#v", call)
	}
	wantUsage := llm.Usage{InputTokens: 17, CachedInputTokens: 4, CacheWriteInputTokens: 3, OutputTokens: 8, ReasoningTokens: 5, Raw: converted.Usage.Raw}
	if !reflect.DeepEqual(converted.Usage, wantUsage) || len(converted.Usage.Raw) == 0 {
		t.Fatalf("usage = %#v", converted.Usage)
	}

	for stop, want := range map[anthropicsdk.StopReason]llm.StopReason{
		anthropicsdk.StopReasonEndTurn:                    llm.StopComplete,
		anthropicsdk.StopReasonStopSequence:               llm.StopComplete,
		anthropicsdk.StopReasonToolUse:                    llm.StopComplete,
		anthropicsdk.StopReasonMaxTokens:                  llm.StopMaxOutputTokens,
		anthropicsdk.StopReasonModelContextWindowExceeded: llm.StopMaxOutputTokens,
		anthropicsdk.StopReasonRefusal:                    llm.StopRefused,
	} {
		if got, err := responseStop(stop); err != nil || got != want {
			t.Errorf("responseStop(%q) = %q, %v; want %q", stop, got, err, want)
		}
	}
	if _, err := responseStop(anthropicsdk.StopReasonPauseTurn); err == nil {
		t.Error("pause_turn was silently treated as a terminal response")
	}
}
