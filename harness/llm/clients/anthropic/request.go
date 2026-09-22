package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/unreallabsai/unreal-agent/harness/llm"
)

func requestParams(request llm.Request, options llm.RequestOptions) (anthropicsdk.MessageNewParams, error) {
	model := strings.TrimSpace(request.Model.ID)
	if model == "" {
		return anthropicsdk.MessageNewParams{}, errors.New("Anthropic model must be set")
	}
	maxTokens := DefaultMaxOutputTokens
	if request.Model.MaxOutputTokens != nil {
		maxTokens = *request.Model.MaxOutputTokens
	}
	if maxTokens <= 0 {
		return anthropicsdk.MessageNewParams{}, errors.New("Anthropic max output tokens must be positive")
	}

	messages, system, err := requestMessages(request.Input)
	if err != nil {
		return anthropicsdk.MessageNewParams{}, err
	}
	tools, err := requestTools(request.Tools)
	if err != nil {
		return anthropicsdk.MessageNewParams{}, err
	}
	params := anthropicsdk.MessageNewParams{
		MaxTokens: maxTokens,
		Messages:  messages,
		Model:     anthropicsdk.Model(model),
		System:    system,
		Tools:     tools,
	}
	if options.CacheKey != "" {
		params.CacheControl = anthropicsdk.NewCacheControlEphemeralParam()
	}
	if request.Model.ReasoningEffort != "" {
		if !request.Model.ReasoningEffort.Valid() {
			return anthropicsdk.MessageNewParams{}, fmt.Errorf("unsupported reasoning effort %q", request.Model.ReasoningEffort)
		}
		if request.Model.ReasoningEffort == llm.ReasoningEffortXHigh && !modelSupportsXHigh(model) {
			return anthropicsdk.MessageNewParams{}, fmt.Errorf("Anthropic model %q does not support xhigh reasoning effort", model)
		}
		params.OutputConfig.Effort = anthropicsdk.OutputConfigEffort(request.Model.ReasoningEffort)
		if modelUsesAdaptiveThinking(model) {
			params.Thinking = anthropicsdk.ThinkingConfigParamUnion{
				OfAdaptive: &anthropicsdk.ThinkingConfigAdaptiveParam{},
			}
		}
	}
	return params, nil
}

func modelSupportsXHigh(model string) bool {
	return modelInFamily(model,
		"claude-fable-5",
		"claude-mythos-5",
		"claude-opus-5",
		"claude-opus-4-8",
		"claude-opus-4-7",
		"claude-sonnet-5",
	)
}

func modelUsesAdaptiveThinking(model string) bool {
	return modelInFamily(model,
		"claude-opus-4-8",
		"claude-opus-4-7",
		"claude-opus-4-6",
		"claude-sonnet-4-6",
	)
}

func modelInFamily(model string, families ...string) bool {
	model = strings.ToLower(strings.TrimSpace(model))
	for _, family := range families {
		if model == family || strings.HasPrefix(model, family+"-") {
			return true
		}
	}
	return false
}

func requestMessages(items []llm.Item) ([]anthropicsdk.MessageParam, []anthropicsdk.TextBlockParam, error) {
	var messages []anthropicsdk.MessageParam
	var system []anthropicsdk.TextBlockParam
	toolCalls := make(map[string]int)
	deliveredResults := make(map[string]struct{})
	for index, item := range items {
		if item.Type == llm.ItemToolResult {
			result, ok := item.Data.(llm.ToolResult)
			if !ok {
				return nil, nil, fmt.Errorf("input item %d: tool result data must be llm.ToolResult, got %T", index, item.Data)
			}
			callMessage, knownCall := toolCalls[result.CallID]
			_, alreadyDelivered := deliveredResults[result.CallID]
			if knownCall && !alreadyDelivered {
				block, err := requestToolResult(result)
				if err != nil {
					return nil, nil, fmt.Errorf("input item %d: %w", index, err)
				}
				var placed bool
				messages, placed = placeToolResult(messages, callMessage, block)
				if placed {
					deliveredResults[result.CallID] = struct{}{}
					continue
				}
			}
			blocks, err := requestToolResultUpdate(result)
			if err != nil {
				return nil, nil, fmt.Errorf("input item %d: %w", index, err)
			}
			messages = appendMessageBlocks(messages, anthropicsdk.MessageParamRoleUser, blocks...)
			continue
		}
		role, block, systemBlock, err := requestItem(item)
		if err != nil {
			return nil, nil, fmt.Errorf("input item %d: %w", index, err)
		}
		if systemBlock != nil {
			system = append(system, *systemBlock)
			continue
		}
		messages = appendMessageBlocks(messages, role, block)
		if item.Type == llm.ItemToolCall {
			toolCalls[item.Data.(llm.ToolCall).CallID] = len(messages) - 1
		}
	}
	return messages, system, nil
}

func placeToolResult(
	messages []anthropicsdk.MessageParam,
	callMessage int,
	block anthropicsdk.ContentBlockParamUnion,
) ([]anthropicsdk.MessageParam, bool) {
	preceding := len(messages) - 1
	if preceding >= 0 && messages[preceding].Role == anthropicsdk.MessageParamRoleUser {
		preceding--
	}
	if preceding != callMessage || preceding < 0 || messages[preceding].Role != anthropicsdk.MessageParamRoleAssistant {
		return messages, false
	}
	if preceding == len(messages)-1 {
		return appendMessageBlocks(messages, anthropicsdk.MessageParamRoleUser, block), true
	}
	content := messages[len(messages)-1].Content
	position := 0
	for position < len(content) && content[position].OfToolResult != nil {
		position++
	}
	content = append(content, anthropicsdk.ContentBlockParamUnion{})
	copy(content[position+1:], content[position:])
	content[position] = block
	messages[len(messages)-1].Content = content
	return messages, true
}

func appendMessageBlocks(messages []anthropicsdk.MessageParam, role anthropicsdk.MessageParamRole, blocks ...anthropicsdk.ContentBlockParamUnion) []anthropicsdk.MessageParam {
	if len(messages) != 0 && messages[len(messages)-1].Role == role {
		messages[len(messages)-1].Content = append(messages[len(messages)-1].Content, blocks...)
		return messages
	}
	return append(messages, anthropicsdk.MessageParam{Role: role, Content: blocks})
}

func requestItem(item llm.Item) (anthropicsdk.MessageParamRole, anthropicsdk.ContentBlockParamUnion, *anthropicsdk.TextBlockParam, error) {
	switch item.Type {
	case llm.ItemMessage:
		message, ok := item.Data.(llm.Message)
		if !ok {
			return "", anthropicsdk.ContentBlockParamUnion{}, nil, fmt.Errorf("message data must be llm.Message, got %T", item.Data)
		}
		if message.Role == llm.RoleSystem {
			return "", anthropicsdk.ContentBlockParamUnion{}, &anthropicsdk.TextBlockParam{Text: message.Text}, nil
		}
		role, err := requestRole(message.Role)
		return role, anthropicsdk.NewTextBlock(message.Text), nil, err

	case llm.ItemToolCall:
		call, ok := item.Data.(llm.ToolCall)
		if !ok {
			return "", anthropicsdk.ContentBlockParamUnion{}, nil, fmt.Errorf("tool call data must be llm.ToolCall, got %T", item.Data)
		}
		arguments := make(map[string]any)
		if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil || arguments == nil {
			arguments = map[string]any{"invalid_arguments": call.Arguments}
		}
		return anthropicsdk.MessageParamRoleAssistant, anthropicsdk.NewToolUseBlock(call.CallID, arguments, call.Name), nil, nil

	case llm.ItemToolResult:
		result, ok := item.Data.(llm.ToolResult)
		if !ok {
			return "", anthropicsdk.ContentBlockParamUnion{}, nil, fmt.Errorf("tool result data must be llm.ToolResult, got %T", item.Data)
		}
		block, err := requestToolResult(result)
		return anthropicsdk.MessageParamRoleUser, block, nil, err

	case llm.ItemReasoning:
		reasoning, ok := item.Data.(llm.Reasoning)
		if !ok {
			return "", anthropicsdk.ContentBlockParamUnion{}, nil, fmt.Errorf("reasoning data must be llm.Reasoning, got %T", item.Data)
		}
		block, err := requestReasoning(reasoning)
		return anthropicsdk.MessageParamRoleAssistant, block, nil, err

	default:
		return "", anthropicsdk.ContentBlockParamUnion{}, nil, fmt.Errorf("unsupported input item type %q", item.Type)
	}
}

func requestRole(role llm.Role) (anthropicsdk.MessageParamRole, error) {
	switch role {
	case llm.RoleUser:
		return anthropicsdk.MessageParamRoleUser, nil
	case llm.RoleAssistant:
		return anthropicsdk.MessageParamRoleAssistant, nil
	default:
		return "", fmt.Errorf("unsupported message role %q", role)
	}
}

func requestReasoning(reasoning llm.Reasoning) (anthropicsdk.ContentBlockParamUnion, error) {
	if len(reasoning.Raw) == 0 {
		return anthropicsdk.ContentBlockParamUnion{}, errors.New("reasoning item must carry the Anthropic content block in Raw")
	}
	var kind struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(reasoning.Raw, &kind); err != nil {
		return anthropicsdk.ContentBlockParamUnion{}, fmt.Errorf("decode reasoning content block: %w", err)
	}
	if kind.Type != "thinking" && kind.Type != "redacted_thinking" {
		return anthropicsdk.ContentBlockParamUnion{}, fmt.Errorf("unsupported Anthropic reasoning content block type %q", kind.Type)
	}
	return param.Override[anthropicsdk.ContentBlockParamUnion](json.RawMessage(reasoning.Raw)), nil
}

func requestToolResult(result llm.ToolResult) (anthropicsdk.ContentBlockParamUnion, error) {
	content := make([]anthropicsdk.ToolResultBlockParamContentUnion, 0, len(result.Output))
	for index, part := range result.Output {
		switch part.Kind {
		case llm.ToolResultText:
			content = append(content, anthropicsdk.ToolResultBlockParamContentUnion{
				OfText: &anthropicsdk.TextBlockParam{Text: part.Value},
			})
		case llm.ToolResultImage:
			image, err := requestImage(part.Value)
			if err != nil {
				return anthropicsdk.ContentBlockParamUnion{}, fmt.Errorf("tool result output %d: %w", index, err)
			}
			content = append(content, anthropicsdk.ToolResultBlockParamContentUnion{OfImage: &image})
		default:
			return anthropicsdk.ContentBlockParamUnion{}, fmt.Errorf("unsupported tool result kind %q", part.Kind)
		}
	}
	block := anthropicsdk.ToolResultBlockParam{ToolUseID: result.CallID, Content: content}
	return anthropicsdk.ContentBlockParamUnion{OfToolResult: &block}, nil
}

func requestToolResultUpdate(result llm.ToolResult) ([]anthropicsdk.ContentBlockParamUnion, error) {
	blocks := []anthropicsdk.ContentBlockParamUnion{anthropicsdk.NewTextBlock(fmt.Sprintf(
		"Follow-up result for tool call %q; the original tool result was already delivered:",
		result.CallID,
	))}
	for index, part := range result.Output {
		switch part.Kind {
		case llm.ToolResultText:
			blocks = append(blocks, anthropicsdk.NewTextBlock(part.Value))
		case llm.ToolResultImage:
			image, err := requestImage(part.Value)
			if err != nil {
				return nil, fmt.Errorf("tool result update output %d: %w", index, err)
			}
			blocks = append(blocks, anthropicsdk.ContentBlockParamUnion{OfImage: &image})
		default:
			return nil, fmt.Errorf("unsupported tool result kind %q", part.Kind)
		}
	}
	return blocks, nil
}

func requestImage(value string) (anthropicsdk.ImageBlockParam, error) {
	header, data, found := strings.Cut(value, ",")
	mediaType, encoding, validHeader := strings.Cut(strings.TrimPrefix(header, "data:"), ";")
	if !found || !strings.HasPrefix(header, "data:") || !validHeader || encoding != "base64" {
		return anthropicsdk.ImageBlockParam{}, errors.New("image output must be a base64 data URL")
	}
	switch mediaType {
	case "image/jpeg", "image/png", "image/gif", "image/webp":
	default:
		return anthropicsdk.ImageBlockParam{}, fmt.Errorf("unsupported image media type %q", mediaType)
	}
	if _, err := base64.StdEncoding.DecodeString(data); err != nil {
		return anthropicsdk.ImageBlockParam{}, errors.New("image output contains invalid base64 data")
	}
	return anthropicsdk.ImageBlockParam{Source: anthropicsdk.ImageBlockParamSourceUnion{
		OfBase64: &anthropicsdk.Base64ImageSourceParam{
			Data: data, MediaType: anthropicsdk.Base64ImageSourceMediaType(mediaType),
		},
	}}, nil
}

func requestTools(tools []llm.Tool) ([]anthropicsdk.ToolUnionParam, error) {
	converted := make([]anthropicsdk.ToolUnionParam, 0, len(tools))
	for index, source := range tools {
		if source.Type != llm.ToolFunction {
			return nil, fmt.Errorf("tool %d: unsupported tool type %q", index, source.Type)
		}
		parameters := source.Parameters
		if parameters == nil {
			parameters = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		encoded, err := json.Marshal(parameters)
		if err != nil {
			return nil, fmt.Errorf("tool %d: encode input schema: %w", index, err)
		}
		schema := param.Override[anthropicsdk.ToolInputSchemaParam](json.RawMessage(encoded))
		tool := anthropicsdk.ToolUnionParamOfTool(schema, source.Name)
		if source.Description != "" {
			tool.OfTool.Description = anthropicsdk.String(source.Description)
		}
		converted = append(converted, tool)
	}
	return converted, nil
}
