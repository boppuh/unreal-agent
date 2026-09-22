package anthropic

import "strings"

// EnvironmentConfig applies the runner's generic API-key override, then the
// Anthropic SDK's environment credential precedence. When it returns no inline
// credential, the SDK can load an Anthropic CLI profile or workload identity.
func EnvironmentConfig(getenv func(string) string) Config {
	if value := strings.TrimSpace(getenv("UNREAL_HARNESS_LLM_API_KEY")); value != "" {
		return Config{APIKey: value}
	}
	if value := strings.TrimSpace(getenv("ANTHROPIC_API_KEY")); value != "" {
		return Config{APIKey: value}
	}
	if value := strings.TrimSpace(getenv("ANTHROPIC_AUTH_TOKEN")); value != "" {
		return Config{AuthToken: value}
	}
	return Config{}
}

func SubscriptionEnvironmentConfig(getenv func(string) string) Config {
	return Config{AuthToken: strings.TrimSpace(getenv("CLAUDE_CODE_OAUTH_TOKEN"))}
}
