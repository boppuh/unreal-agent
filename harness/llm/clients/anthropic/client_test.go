package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/unreallabsai/unreal-agent/harness/llm"
)

func TestClientStreamsMessage(t *testing.T) {
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "ambient-auth-token")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/messages" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("X-Api-Key") != "api-key" || r.Header.Get("Authorization") != "" ||
			r.Header.Get("anthropic-version") == "" {
			t.Errorf("headers = %#v", r.Header)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body["stream"] != true || body["max_tokens"] != float64(DefaultMaxOutputTokens) {
			t.Errorf("body = %#v", body)
		}
		writeMessageSSE(w, "subscription works")
	}))
	defer server.Close()
	client, err := NewClient(Config{APIKey: "api-key", BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	response, err := client.Respond(t.Context(), llm.Request{
		Model: llm.Model{ID: "claude-test"},
		Input: []llm.Item{{Type: llm.ItemMessage, Data: llm.Message{Role: llm.RoleUser, Text: "hello"}}},
	}, llm.RequestOptions{})
	if err != nil || response.ID != "msg_1" || response.Stop != llm.StopComplete || len(response.Output) != 1 ||
		response.Output[0].Data.(llm.Message).Text != "subscription works" {
		t.Fatalf("response = %#v, error = %v", response, err)
	}
	var rawUsage struct {
		OutputTokens int64 `json:"output_tokens"`
	}
	if err := json.Unmarshal(response.Usage.Raw, &rawUsage); err != nil || response.Usage.OutputTokens != 3 || rawUsage.OutputTokens != 3 {
		t.Fatalf("usage = %#v, raw = %s, error = %v", response.Usage, response.Usage.Raw, err)
	}
}

func TestClientPreservesStreamedThinkingSignature(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range []struct {
			name, data string
		}{
			{"message_start", `{"type":"message_start","message":{"id":"msg_thinking","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":2,"output_tokens":0}}}`},
			{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"inspect first"}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"opaque-signature"}}`},
			{"content_block_stop", `{"type":"content_block_stop","index":0}`},
			{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"Bash","input":{}}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"command\":\"pwd\"}"}}`},
			{"content_block_stop", `{"type":"content_block_stop","index":1}`},
			{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":9,"output_tokens_details":{"thinking_tokens":4}}}`},
			{"message_stop", `{"type":"message_stop"}`},
		} {
			_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event.name, event.data)
		}
	}))
	defer server.Close()
	client, err := NewClient(Config{APIKey: "api-key", BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	response, err := client.Respond(t.Context(), llm.Request{
		Model: llm.Model{ID: "claude-test"},
		Input: []llm.Item{{Type: llm.ItemMessage, Data: llm.Message{Role: llm.RoleUser, Text: "hello"}}},
	}, llm.RequestOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Output) != 2 || response.Output[0].Type != llm.ItemReasoning || response.Output[1].Type != llm.ItemToolCall {
		t.Fatalf("response output = %#v", response.Output)
	}
	reasoning := response.Output[0].Data.(llm.Reasoning)
	if !strings.Contains(string(reasoning.Raw), `"thinking":"inspect first"`) ||
		!strings.Contains(string(reasoning.Raw), `"signature":"opaque-signature"`) {
		t.Fatalf("reasoning raw = %s", reasoning.Raw)
	}
	if response.Usage.OutputTokens != 9 || response.Usage.ReasoningTokens != 4 {
		t.Fatalf("usage = %#v", response.Usage)
	}
}

func TestStandardClientRejectsCrossOriginRedirects(t *testing.T) {
	var redirected atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, request *http.Request) {
		redirected.Add(1)
		if request.Header.Get("X-Api-Key") != "" {
			t.Error("API key reached cross-origin redirect target")
		}
	}))
	defer target.Close()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Api-Key") != "api-key" {
			t.Errorf("source headers = %#v", r.Header)
		}
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer server.Close()
	client, err := NewClient(Config{APIKey: "api-key", BaseURL: server.URL, MaxAttempts: new(1)})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	_, err = client.Respond(t.Context(), llm.Request{
		Model: llm.Model{ID: "claude-test"},
		Input: []llm.Item{{Type: llm.ItemMessage, Data: llm.Message{Role: llm.RoleUser, Text: "hello"}}},
	}, llm.RequestOptions{})
	if err == nil || redirected.Load() != 0 {
		t.Fatalf("error = %v, redirected requests = %d", err, redirected.Load())
	}
}

func TestStandardClientAllowsSameOriginRedirects(t *testing.T) {
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Api-Key") != "api-key" {
			t.Errorf("headers = %#v", r.Header)
		}
		switch r.URL.Path {
		case "/v1/messages":
			http.Redirect(w, r, server.URL+"/redirected", http.StatusTemporaryRedirect)
		case "/redirected":
			writeMessageSSE(w, "same origin")
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	client, err := NewClient(Config{APIKey: "api-key", BaseURL: server.URL, MaxAttempts: new(1)})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	response, err := client.Respond(t.Context(), llm.Request{
		Model: llm.Model{ID: "claude-test"},
		Input: []llm.Item{{Type: llm.ItemMessage, Data: llm.Message{Role: llm.RoleUser, Text: "hello"}}},
	}, llm.RequestOptions{})
	if err != nil || len(response.Output) != 1 || response.Output[0].Data.(llm.Message).Text != "same origin" {
		t.Fatalf("response = %#v, error = %v", response, err)
	}
}

func TestSubscriptionClientUsesOAuthHeadersAndRejectsRedirects(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "ambient-api-key")
	var redirected atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { redirected.Add(1) }))
	defer target.Close()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer oauth-token" || r.Header.Get("anthropic-beta") != subscriptionBeta ||
			r.Header.Get("X-Api-Key") != "" {
			t.Errorf("headers = %#v", r.Header)
		}
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer server.Close()
	client, err := NewSubscriptionClient(Config{AuthToken: "oauth-token", BaseURL: server.URL, MaxAttempts: new(1)})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	_, err = client.Respond(t.Context(), llm.Request{
		Model: llm.Model{ID: "claude-test"},
		Input: []llm.Item{{Type: llm.ItemMessage, Data: llm.Message{Role: llm.RoleUser, Text: "hello"}}},
	}, llm.RequestOptions{})
	if err == nil || redirected.Load() != 0 {
		t.Fatalf("error = %v, redirected requests = %d", err, redirected.Load())
	}
}

func TestSubscriptionClientWrapsUnauthorizedAndHonorsCancellation(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"authentication_error","message":"bad token"}}`)
	}))
	defer server.Close()
	client, err := NewSubscriptionClient(Config{AuthToken: "oauth-token", BaseURL: server.URL, MaxAttempts: new(1)})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	request := llm.Request{Model: llm.Model{ID: "claude-test"}, Input: []llm.Item{{Type: llm.ItemMessage, Data: llm.Message{Role: llm.RoleUser, Text: "hello"}}}}
	if _, err := client.Respond(t.Context(), request, llm.RequestOptions{}); err == nil || !strings.Contains(err.Error(), "renew CLAUDE_CODE_OAUTH_TOKEN") {
		t.Fatalf("error = %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := client.Respond(ctx, request, llm.RequestOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled response error = %v", err)
	}
	if requests.Load() != 1 {
		t.Fatalf("requests = %d", requests.Load())
	}
}

func TestClientConfiguration(t *testing.T) {
	for _, url := range []string{"", BaseURL, BaseURL + "/", "http://127.0.0.1:1234", "http://[::1]:1234"} {
		if _, err := validateBaseURL(url, true); err != nil {
			t.Errorf("subscription rejected %q: %v", url, err)
		}
	}
	for _, url := range []string{"https://example.com", "http://api.anthropic.com", "https://api.anthropic.com.evil.example", BaseURL + "?token=bad", "http://localhost:1234", "http://user:pass@127.0.0.1:1234"} {
		if _, err := validateBaseURL(url, true); err == nil {
			t.Errorf("subscription accepted %q", url)
		}
	}
	for _, config := range []Config{
		{APIKey: "key", AuthToken: "token"},
		{AuthToken: "token", MaxAttempts: new(0)},
		{AuthToken: "bad\ntoken"},
	} {
		if client, err := NewSubscriptionClient(config); err == nil || client != nil {
			t.Fatalf("invalid subscription config accepted: %#v", config)
		}
	}
	if client, err := NewClient(Config{BaseURL: "https://gateway.example/v1"}); err != nil || client == nil {
		t.Fatalf("standard custom endpoint rejected: %v", err)
	}
	if client, err := NewClient(Config{APIKey: "key", BaseURL: "http://gateway.example/v1"}); err == nil || client != nil {
		t.Fatalf("standard remote HTTP endpoint accepted: %#v, %v", client, err)
	}
}

func TestEnvironmentConfigs(t *testing.T) {
	getenv := func(name string) string {
		return map[string]string{
			"UNREAL_HARNESS_LLM_API_KEY": " generic ",
			"ANTHROPIC_API_KEY":          "provider",
			"ANTHROPIC_AUTH_TOKEN":       "bearer",
			"CLAUDE_CODE_OAUTH_TOKEN":    " oauth ",
		}[name]
	}
	if got := EnvironmentConfig(getenv); got.APIKey != "generic" || got.AuthToken != "" {
		t.Fatalf("standard config = %#v", got)
	}
	if got := SubscriptionEnvironmentConfig(getenv); got.AuthToken != "oauth" || got.APIKey != "" {
		t.Fatalf("subscription config = %#v", got)
	}
	if got := EnvironmentConfig(func(string) string { return "" }); got.APIKey != "" || got.AuthToken != "" {
		t.Fatalf("profile config = %#v", got)
	}
}

func writeMessageSSE(w http.ResponseWriter, text string) {
	w.Header().Set("Content-Type", "text/event-stream")
	events := []struct {
		name, data string
	}{
		{"message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":2,"output_tokens":0}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":%q}}`, text)},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
	for _, event := range events {
		_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event.name, event.data)
	}
}
