package agent

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"sync/atomic"

	"github.com/go-logr/logr"

	"github.com/agent-api/core"
	"github.com/agent-api/core/agent/bootstrap"
	"github.com/agent-api/core/memory/array"
)

// Agent represents a basic AI agent with its configuration and state
type Agent struct {
	// interfaces
	mem          core.MemoryBackend
	provider     core.Provider
	vecStore     core.VectorStorer
	systemPrompt string

	tools ToolMap

	maxSteps int

	logger *logr.Logger
}

// NewAgent creates a new agent with the given provider
func NewAgent(opts ...bootstrap.NewAgentConfigFunc) (*Agent, error) {
	conf := &bootstrap.NewAgentConfig{
		Provider:     nil,
		MaxSteps:     25,
		Tools:        []*core.Tool{},
		SystemPrompt: "You are a helpful assistant",
		Logger:       nil,
		Memory:       nil,
	}

	// Apply all option functions
	for _, opt := range opts {
		opt(conf)
	}

	// no provider set, return err
	if conf.Provider == nil {
		return nil, fmt.Errorf("no provider set")
	}

	// set a default logger if non provided
	if conf.Logger == nil {
		slogOpts := slog.HandlerOptions{
			AddSource: true,
			Level:     slog.Level(-1),
		}
		handler := slog.NewTextHandler(os.Stderr, &slogOpts)
		l := logr.FromSlogHandler(handler)
		conf.Logger = &l
	}

	// set the default memory if non provided
	if conf.Memory == nil {
		conf.Memory = array.NewArrayMemoryBackend()
	}

	agent := &Agent{
		provider:     conf.Provider,
		tools:        make(map[string]*core.Tool),
		vecStore:     conf.VecStore,
		mem:          conf.Memory,
		maxSteps:     conf.MaxSteps,
		logger:       conf.Logger,
		systemPrompt: conf.SystemPrompt,
	}

	// set tools
	for _, tool := range conf.Tools {
		agent.tools[tool.Name] = tool
	}

	if agent.vecStore != nil {
		agent.addVecStoreTools()
	}

	return agent, nil
}

func (a *Agent) addVecStoreTools() {
	// TODO - this json schema is, unfortunately, not valid for
	jsonSchema := `{
      "title": "searchVectorStore",
      "description": "Searches a connected vector store for relevant and related content",
      "type": "object",
      "properties": {
        "Query": {
          "description": "The query to use on the vector store",
          "type": "string"
        },
        "Limit": {
          "description": "The limit of returned content",
          "type": "number"
        }
      },
      "required": [
        "Query"
      ]
    }`

	// Register the vector searching functionality
	wrappedSearch, err := core.WrapToolFunction(a.vecStore.Search)
	if err != nil {
		panic(err)
	}

	err = a.AddTool(&core.Tool{
		Name:                "searchVectorStore",
		Description:         "Searches the connected vector store for similar, related content",
		WrappedToolFunction: wrappedSearch,
		JSONSchema:          []byte(jsonSchema),
	})
	if err != nil {
		panic(err)
	}
}

// Run implements the main agent loop
func (a *Agent) Run(ctx context.Context, opts ...RunOptionFunc) (*AgentRunAggregator, error) {
	// Initialize with default options
	runOpts := &RunOptions{
		Input:         "Execute given tasks.",
		StopCondition: DefaultStopCondition,
		Images:        []*core.Image{},
	}

	// Apply all option functions
	for _, opt := range opts {
		opt(runOpts)
	}

	var id uint32 = 0

	a.prepareMemory(id)

	agg := NewAgentRunAggregator()
	m := &core.Message{
		ID:         id,
		Role:       core.UserMessageRole,
		Content:    runOpts.Input,
		Images:     runOpts.Images,
		ToolCalls:  nil,
		ToolResult: nil,
		Metadata:   nil,
	}
	agg.Push(m)

	err := a.mem.Add(m)
	if err != nil {
		panic(err)
	}

	for {
		a.logger.V(1).Info("retrieving messages from memory backend")
		messages, err := a.mem.GetMaxN(10)
		if err != nil {
			panic(err)
		}

		a.logger.V(1).Info("sending messages", "messages", messages)

		respMessage, respErr := a.SendMessages(ctx, messages)
		agg.Push(respMessage)
		if respErr != nil {
			return agg, respErr
		}

		respMessage.ID = atomic.AddUint32(&id, 1)
		a.logger.V(1).Info("response message", "message", respMessage)

		// Add to memory
		err = a.mem.Add(respMessage)
		if err != nil {
			panic(err)
		}

		// Check stop condition
		if runOpts.StopCondition(agg) {
			a.logger.V(1).Info("reached stop condition", "steps", len(agg.Messages))
			return agg, nil
		}

		// Check max steps
		if len(agg.Messages) >= a.maxSteps {
			a.logger.V(-1).Info("exceeded max steps", "steps", len(agg.Messages))
			return agg, fmt.Errorf("exceeded maximum steps: %d - %d", len(agg.Messages), a.maxSteps)
		}

		// 2 "send" scenarios:
		//    * "user" message
		//    * "tool" results message
		//
		// 1 "receive" scenario:
		//    * LLM responds with "content" and "tool_calls". Either or may be empty

		// Call tools if tool calls were present
		if len(respMessage.ToolCalls) > 0 {
			toolResponses := a.executeToolCallsParallel(ctx, respMessage.ToolCalls, id)
			agg.Push(toolResponses...)
			a.mem.Add(toolResponses...)
		}
	}
}

func (a *Agent) prepareMemory(id uint32) {
	messages, err := a.mem.GetMaxN(1)
	if err != nil {
		panic(err)
	}
	if len(messages) > 0 {
		return
	}

	sysM := &core.Message{
		ID:         id,
		Role:       core.SystemMessageRole,
		Content:    a.systemPrompt,
		Images:     nil,
		ToolCalls:  nil,
		ToolResult: nil,
		Metadata:   nil,
	}
	err = a.mem.Add(sysM)
	if err != nil {
		panic(err)
	}
}

type StreamRunnerResults struct {
	AggChan   <-chan AgentRunAggregator
	DeltaChan <-chan string
	ErrChan   <-chan error
}

// RunStream supports a streaming channel from a provider
func (a *Agent) RunStream(ctx context.Context, opts ...RunOptionFunc) *StreamRunnerResults {
	// Initialize with default options
	runOpts := &RunOptions{
		Input:         "Execute given tasks.",
		StopCondition: DefaultStopCondition,
		Images:        []*core.Image{},
	}

	// Apply all option functions
	for _, opt := range opts {
		opt(runOpts)
	}

	var id uint32 = 0

	// buffered, non-blocking channels
	outAggChan := make(chan AgentRunAggregator, 10)
	outDeltaChan := make(chan string, 10)
	outErrChan := make(chan error, 10)

	result := &StreamRunnerResults{
		AggChan:   outAggChan,
		DeltaChan: outDeltaChan,
		ErrChan:   outErrChan,
	}

	// init aggregator
	agg := NewAgentRunAggregator()
	m := &core.Message{
		Role:       core.UserMessageRole,
		Content:    runOpts.Input,
		Images:     runOpts.Images,
		ToolCalls:  nil,
		ToolResult: nil,
		Metadata:   nil,
	}
	agg.Push(nil, m)

	err := a.mem.Add(m)
	if err != nil {
		panic(err)
	}

	a.logger.V(1).Info("kicking run streamer")

	go func() {
		defer close(outAggChan)
		defer close(outDeltaChan)
		defer close(outErrChan)

		// Send initial aggregator state (non-blocking)
		select {
		case outAggChan <- *agg:
		default:
			// Skip if no one is listening
		}

		for {
			// Get streaming response for current messages

			messages, err := a.mem.GetMaxN(10)
			if err != nil {
				panic(err)
			}

			msgChan, deltaChan, errChan := a.SendMessageStream(ctx, messages)

			var respMessage *core.Message
			var respErr error

			for {
				// escape inner loop if we're all done with this message stream
				allClosed := msgChan == nil && deltaChan == nil && errChan == nil
				if allClosed {
					break
				}

				select {
				case msg, ok := <-msgChan:
					if !ok {
						a.logger.V(1).Info("send message message chan closed")
						msgChan = nil
						continue
					}
					if msg != nil {
						a.logger.Info("received message",
							"role", msg.Role,
							"content", msg.Content,
							"tool_calls", msg.ToolCalls,
						)
						respMessage = msg
						respMessage.ID = atomic.AddUint32(&id, 1)
					}

				case delta, ok := <-deltaChan:
					if !ok {
						a.logger.V(1).Info("send message delta chan closed")
						deltaChan = nil
						continue
					}

					if delta != "" {
						select {
						case outDeltaChan <- delta:
						default:
							// Skip if no one is listening
						}
					}

					// pull errors from the downstream provider error channel.
				case err, ok := <-errChan:
					if !ok {
						a.logger.V(1).Info("send message err chan closed")
						errChan = nil
						continue
					}
					if err != nil {
						respErr = err
						// Forward error to output channel (non-blocking)
						select {
						case outErrChan <- err:
						default:
							// Skip if no one is listening
						}
					}

				case <-ctx.Done():
					select {
					case outErrChan <- ctx.Err():
					default:
						// Skip if no one is listening
					}

					return
				}
			}

			// If we got a response message, add it to the aggregator
			if respMessage != nil {
				agg.Push(respMessage)
				a.mem.Add(respMessage)

				select {
				case outAggChan <- *agg:
				default:
					// Skip if no one is listening
				}
			}

			// If there was an error, return
			if respErr != nil {
				return
			}

			// Check stop condition
			if runOpts.StopCondition(agg) {
				a.logger.V(1).Info("reached stop condition", "steps", len(agg.Messages))
				return
			}

			// Check max steps
			if len(agg.Messages) >= a.maxSteps {
				respErr = fmt.Errorf("exceeded maximum steps: %d - %d", len(agg.Messages), a.maxSteps)

				select {
				case outErrChan <- respErr:
				default:
					// Skip if no one is listening
				}
				return
			}

			// Call tools if tool calls were present
			if respMessage != nil && len(respMessage.ToolCalls) > 0 {
				toolResponses := a.executeToolCallsParallel(ctx, respMessage.ToolCalls, id)
				agg.Push(toolResponses...)
				a.mem.Add(toolResponses...)

				// Send updated aggregator after tool execution
				select {
				case outAggChan <- *agg:
				default:
					// Skip if no one is listening
				}
			}
		}
	}()

	return result
}

// SendMessage sends a message to the agent and gets a response
func (a *Agent) SendMessages(ctx context.Context, m []*core.Message) (*core.Message, error) {
	toolSlice := make([]*core.Tool, 0, len(a.tools))
	for _, tool := range a.tools {
		toolSlice = append(toolSlice, tool)
	}

	genOpts := &core.GenerateOptions{
		Messages: m,
		Tools:    toolSlice,
	}

	a.logger.V(1).Info("sending message with generate options", "genOpts", genOpts)
	response, err := a.provider.Generate(ctx, genOpts)
	if err != nil {
		return nil, err
	}

	return response, nil
}

// SendMessage sends a message to the agent and gets a response
func (a *Agent) SendMessageStream(ctx context.Context, m []*core.Message) (<-chan *core.Message, <-chan string, <-chan error) {
	toolSlice := make([]*core.Tool, 0, len(a.tools))
	for _, tool := range a.tools {
		toolSlice = append(toolSlice, tool)
	}

	genOpts := &core.GenerateOptions{
		Messages: m,
		Tools:    toolSlice,
	}

	a.logger.V(1).Info("sending message with generate options", "genOpts", genOpts)
	return a.provider.GenerateStream(ctx, genOpts)
}

// CallTool sends a message to the agent and gets a response
func (a *Agent) CallTool(ctx context.Context, tc *core.ToolCall) (*core.Message, error) {
	// Find the corresponding tool
	var toolToCall *core.Tool

	for _, t := range a.tools {
		if t.Name == tc.Name {
			toolToCall = t
			break
		}
	}

	if toolToCall == nil {
		return nil, fmt.Errorf("tool %s not found", tc.Name)
	}

	// Call the tool
	result, err := toolToCall.WrappedToolFunction(ctx, []byte(tc.Arguments))
	if err != nil {
		return nil, fmt.Errorf("tool execution failed: %w", err)
	}

	// Add the tool response to messages
	return &core.Message{
		Role:    core.ToolMessageRole,
		Content: fmt.Sprintf("%v", result),
		ToolResult: []*core.ToolResult{
			{
				ToolCallID: tc.ID,
				Content:    result,
				Error:      "",
			},
		},
	}, nil
}

// AddTool adds a tool to the agent's available tools
func (a *Agent) AddTool(tool *core.Tool) error {
	if tool.Name == "" {
		return errors.New("tool must have a name")
	}

	if tool.WrappedToolFunction == nil {
		return errors.New("tool must have a function")
	}

	a.tools[tool.Name] = tool

	return nil
}

// Example stop condition
func DefaultStopCondition(agg *AgentRunAggregator) bool {
	// Stop if no tool calls were made and we got a response
	if len(agg.Messages) != 0 {
		if len(agg.Messages[len(agg.Messages)-1].ToolCalls) == 0 && len(agg.Messages[len(agg.Messages)-1].Content) != 0 {
			return true
		}
	}

	return false
}

// executeToolCallsParallel executes multiple tool calls in parallel using WaitGroup
func (a *Agent) executeToolCallsParallel(ctx context.Context, toolCalls []*core.ToolCall, id uint32) []*core.Message {
	var wg sync.WaitGroup
	responses := make([]*core.Message, len(toolCalls))

	for i, toolCall := range toolCalls {
		wg.Add(1)

		// Launch each tool call in its own goroutine
		go func(i int, tc *core.ToolCall) {
			defer wg.Done()

			a.logger.V(1).Info("calling tool", "tool", tc.Name, "id", tc.ID)
			toolResp, internalErr := a.CallTool(ctx, tc)

			// handle the internal tool calling error
			// (this is different from errors related to LLM hallucinations like
			// improperly formatted json or missing required params)
			if internalErr != nil {
				a.logger.V(-1).Info("tool execution failed",
					"tool", tc.Name,
					"error", internalErr)

				toolResp = &core.Message{
					ID:        atomic.AddUint32(&id, 1),
					Role:      core.ToolMessageRole,
					Content:   "",
					ToolCalls: nil,
					ToolResult: []*core.ToolResult{
						{
							ToolCallID: tc.ID,
							Error:      fmt.Sprintf("internal error executing tool %s: %v", tc.Name, internalErr),
						},
					},
					Metadata: nil,
				}
			}

			a.logger.V(1).Info("tool response message", "message", toolResp)
			responses[i] = toolResp
		}(i, toolCall)
	}

	// Wait for all tool calls to complete
	wg.Wait()
	return responses
}
