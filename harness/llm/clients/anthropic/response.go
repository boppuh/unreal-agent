package anthropic

import (
	"encoding/json/jsontext"
	"fmt"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/unreallabsai/unreal-agent/harness/llm"
)

func response(source anthropicsdk.Message) (llm.Response, error) {
	stop, err := responseStop(source.StopReason)
	if err != nil {
		return llm.Response{}, err
	}
	converted := llm.Response{
		ID:    source.ID,
		Stop:  stop,
		Usage: responseUsage(source.Usage),
	}
	converted.Output = make([]llm.Item, 0, len(source.Content))
	for index, block := range source.Content {
		item, err := responseItem(source.ID, index, block)
		if err != nil {
			return llm.Response{}, fmt.Errorf("output content block %d: %w", index, err)
		}
		converted.Output = append(converted.Output, item)
	}
	return converted, nil
}

func responseItem(messageID string, index int, block anthropicsdk.ContentBlockUnion) (llm.Item, error) {
	providerID := fmt.Sprintf("%s:%d", messageID, index)
	switch block.Type {
	case "text":
		return llm.Item{
			ProviderID: providerID,
			Type:       llm.ItemMessage,
			Data:       llm.Message{Role: llm.RoleAssistant, Text: block.Text},
		}, nil
	case "tool_use":
		return llm.Item{
			ProviderID: providerID,
			Type:       llm.ItemToolCall,
			Data: llm.ToolCall{
				CallID: block.ID, Name: block.Name, Arguments: string(block.Input),
			},
		}, nil
	case "thinking", "redacted_thinking":
		raw := jsontext.Value(append([]byte(nil), block.RawJSON()...))
		reasoning := llm.Reasoning{Raw: raw}
		if block.Type == "thinking" && block.Thinking != "" {
			reasoning.Summary = []string{block.Thinking}
		}
		return llm.Item{
			ProviderID: providerID,
			Type:       llm.ItemReasoning,
			Data:       reasoning,
		}, nil
	default:
		return llm.Item{}, fmt.Errorf("unsupported Anthropic content block type %q", block.Type)
	}
}

func responseStop(source anthropicsdk.StopReason) (llm.StopReason, error) {
	switch source {
	case anthropicsdk.StopReasonEndTurn,
		anthropicsdk.StopReasonStopSequence,
		anthropicsdk.StopReasonToolUse:
		return llm.StopComplete, nil
	case anthropicsdk.StopReasonMaxTokens,
		anthropicsdk.StopReasonModelContextWindowExceeded:
		return llm.StopMaxOutputTokens, nil
	case anthropicsdk.StopReasonRefusal:
		return llm.StopRefused, nil
	default:
		return "", fmt.Errorf("unsupported Anthropic stop reason %q", source)
	}
}

func responseUsage(source anthropicsdk.Usage) llm.Usage {
	return llm.Usage{
		InputTokens:           source.InputTokens + source.CacheCreationInputTokens + source.CacheReadInputTokens,
		CachedInputTokens:     source.CacheReadInputTokens,
		CacheWriteInputTokens: source.CacheCreationInputTokens,
		OutputTokens:          source.OutputTokens,
		ReasoningTokens:       source.OutputTokensDetails.ThinkingTokens,
		Raw:                   jsontext.Value(append([]byte(nil), source.RawJSON()...)),
	}
}
