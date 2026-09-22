package anthropic

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/unreallabsai/unreal-agent/harness/llm"
)

const (
	BaseURL                = "https://api.anthropic.com"
	DefaultMaxAttempts     = 5
	DefaultMaxOutputTokens = int64(32_000)
	subscriptionBeta       = "oauth-2025-04-20"
)

type Config struct {
	APIKey      string
	AuthToken   string
	BaseURL     string
	MaxAttempts *int
	HTTPClient  *http.Client
}

type Client struct {
	sdk          anthropicsdk.Client
	httpClient   *http.Client
	subscription bool
}

var _ llm.Adapter = (*Client)(nil)

func NewClient(config Config) (*Client, error) {
	return newClient(config, false)
}

// NewSubscriptionClient uses a Claude Code subscription OAuth token. It only
// sends that token to Anthropic's production API, or to an explicit loopback
// endpoint used for local development and tests, and never follows redirects.
func NewSubscriptionClient(config Config) (*Client, error) {
	return newClient(config, true)
}

func newClient(config Config, subscription bool) (*Client, error) {
	if config.MaxAttempts != nil && *config.MaxAttempts <= 0 {
		return nil, errors.New("max attempts must be positive")
	}
	if strings.TrimSpace(config.APIKey) != "" && strings.TrimSpace(config.AuthToken) != "" {
		return nil, errors.New("Anthropic API key and auth token are mutually exclusive")
	}

	baseURL, err := validateBaseURL(config.BaseURL, subscription)
	if err != nil {
		return nil, err
	}
	apiKey, authToken := strings.TrimSpace(config.APIKey), strings.TrimSpace(config.AuthToken)
	if subscription {
		if apiKey != "" {
			return nil, errors.New("Anthropic subscription authentication does not accept an API key")
		}
		if authToken == "" {
			return nil, errors.New("Anthropic subscription OAuth token must be set in CLAUDE_CODE_OAUTH_TOKEN")
		}
		if !validHeaderValue(authToken) {
			return nil, errors.New("Anthropic subscription OAuth token is not a valid header value")
		}
	}

	maxAttempts := DefaultMaxAttempts
	if config.MaxAttempts != nil {
		maxAttempts = *config.MaxAttempts
	}
	opts := make([]option.RequestOption, 0, 8)
	if subscription || apiKey != "" || authToken != "" {
		// An explicit credential must never be combined with an API key,
		// profile, or workload identity from the ambient credential chain.
		opts = append(opts, option.WithoutEnvironmentDefaults())
	}
	opts = append(opts,
		option.WithBaseURL(baseURL),
		option.WithMaxRetries(maxAttempts-1),
	)
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}
	if authToken != "" {
		opts = append(opts, option.WithAuthToken(authToken))
	}

	httpClient := config.HTTPClient
	if subscription {
		httpClient = noRedirectHTTPClient(httpClient)
		opts = append(opts,
			option.WithHTTPClient(httpClient),
			option.WithHeader("anthropic-beta", subscriptionBeta),
			option.WithHeader("User-Agent", "unreal-agent"),
		)
	} else if httpClient != nil {
		opts = append(opts, option.WithHTTPClient(httpClient))
	}

	return &Client{
		sdk:          anthropicsdk.NewClient(opts...),
		httpClient:   httpClient,
		subscription: subscription,
	}, nil
}

func (client *Client) Respond(ctx context.Context, request llm.Request, options llm.RequestOptions) (llm.Response, error) {
	if err := ctx.Err(); err != nil {
		return llm.Response{}, err
	}
	params, err := requestParams(request, options)
	if err != nil {
		return llm.Response{}, err
	}

	stream := client.sdk.Messages.NewStreaming(ctx, params)
	var message anthropicsdk.Message
	for stream.Next() {
		if err := message.Accumulate(stream.Current()); err != nil {
			_ = stream.Close()
			return llm.Response{}, fmt.Errorf("accumulate Anthropic response: %w", err)
		}
	}
	streamErr := stream.Err()
	closeErr := stream.Close()
	if streamErr != nil || closeErr != nil {
		err := errors.Join(streamErr, closeErr)
		var apiErr *anthropicsdk.Error
		if client.subscription && errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusUnauthorized {
			return llm.Response{}, fmt.Errorf("Anthropic subscription credentials rejected; renew CLAUDE_CODE_OAUTH_TOKEN externally and recreate the client: %w", err)
		}
		return llm.Response{}, fmt.Errorf("stream Anthropic response: %w", err)
	}
	return response(message)
}

func (client *Client) Close() error {
	if client.httpClient != nil {
		client.httpClient.CloseIdleConnections()
	}
	return nil
}

func validateBaseURL(value string, subscription bool) (string, error) {
	value = strings.TrimRight(strings.TrimSpace(value), "/")
	if value == "" {
		value = BaseURL
	}
	parsed, err := url.Parse(value)
	if err != nil || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" || parsed.Hostname() == "" ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		return "", errors.New("Anthropic base URL must be an HTTP(S) endpoint without credentials, query parameters, or a fragment")
	}
	if !subscription {
		return value, nil
	}
	if value == BaseURL {
		return value, nil
	}
	if ip := net.ParseIP(parsed.Hostname()); ip != nil && ip.IsLoopback() {
		return value, nil
	}
	return "", errors.New("Anthropic subscription base URL must be https://api.anthropic.com or an explicit loopback IP endpoint")
}

func noRedirectHTTPClient(source *http.Client) *http.Client {
	if source == nil {
		return &http.Client{
			Transport:     http.DefaultTransport.(*http.Transport).Clone(),
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		}
	}
	copy := *source
	copy.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	return &copy
}

func validHeaderValue(value string) bool {
	for _, char := range value {
		if char <= ' ' || char > '~' {
			return false
		}
	}
	return value != ""
}
