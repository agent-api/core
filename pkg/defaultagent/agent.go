package defaultagent

import (
	"context"
	"errors"
	"fmt"

	"github.com/agent-api/core"
	"github.com/agent-api/core/agent"
	"github.com/agent-api/core/message"
	"github.com/agent-api/core/tool"
)

// DefaultAgent represents a basic AI agent with its configuration and state
type DefaultAgent struct {
	provider core.Provider
	tools    ToolMap
	memory   []message.Message
	config   *agent.AgentConfig
}

type ToolMap map[string]tool.Tool

// NewAgent creates a new agent with the given provider
func NewAgent(config *agent.AgentConfig) *DefaultAgent {
	if config.MaxSteps == 0 {
		// set a sane default max steps
		config.MaxSteps = 5
	}

	return &DefaultAgent{
		provider: config.Provider,
		tools:    make(map[string]tool.Tool),
		memory:   make([]message.Message, 0),
		config:   config,
	}
}

// Run implements the main agent loop
func (a *DefaultAgent) Run(ctx context.Context, input string, stopCondition agent.StopCondition) ([]*agent.AgentStep, error) {
	var steps []*agent.AgentStep

	currentStep := &agent.AgentStep{
		ID: "1",
		Message: &message.Message{
			Role:       message.UserMessageRole,
			Content:    input,
			ToolCalls:  nil,
			ToolResult: nil,
			Metadata:   nil,
		},
		Error: nil,
	}

	steps = append(steps, currentStep)

	for {
		fmt.Printf("Current message: %v\n", currentStep.Message)
		respMessage, err := a.SendMessage(ctx, *currentStep.Message)
		respStep := &agent.AgentStep{
			ID:      "2",
			Message: respMessage,
			Error:   err,
		}
		steps = append(steps, respStep)
		if err != nil {
			return steps, err
		}

		fmt.Printf("Resp message: %v\n", respMessage)

		// Check stop condition
		if stopCondition(respStep) {
			return steps, nil
		}

		// Check max steps
		if len(steps) >= a.config.MaxSteps {
			return steps, fmt.Errorf("exceeded maximum steps: %d - %d", len(steps), a.config.MaxSteps)
		}

		// 2 "send" scenarios:
		//    * "user" message
		//    * "tool" results message
		//
		// 1 "receive" scenario:
		//    * LLM responds with "content" and "tool_calls". Either or may be empty

		// Prepare next input based on tool results
		if len(respStep.Message.ToolCalls) > 0 {
			toolMessage, err := a.CallTool(ctx, respStep.Message.ToolCalls[0])
			fmt.Printf("%v\n", toolMessage)
			currentStep = &agent.AgentStep{
				ID:      "3",
				Message: toolMessage,
				Error:   err,
			}
		} else {
			currentStep = &agent.AgentStep{
				ID:      "4",
				Message: respMessage,
				Error:   err,
			}
		}
	}
}

// SendMessage sends a message to the agent and gets a response
func (a *DefaultAgent) SendMessage(ctx context.Context, m message.Message) (*message.Message, error) {
	a.memory = append(a.memory, m)

	var response *message.Message
	var err error

	toolSlice := make([]tool.Tool, 0, len(a.tools))
	for _, tool := range a.tools {
		toolSlice = append(toolSlice, tool)
	}

	genOpts := &core.GenerateOptions{
		Messages: a.memory,
		Tools:    toolSlice,
	}

	response, err = a.provider.Generate(ctx, genOpts)
	if err != nil {
		return nil, err
	}

	a.memory = append(a.memory, *response)
	return response, nil
}

// CallTool sends a message to the agent and gets a response
func (a *DefaultAgent) CallTool(ctx context.Context, tc message.ToolCall) (*message.Message, error) {
	// Find the corresponding tool
	var toolToCall *tool.Tool

	for _, t := range a.tools {
		if t.Name == tc.Name {
			toolToCall = &t
			break
		}
	}

	if toolToCall == nil {
		return nil, fmt.Errorf("tool %s not found", tc.Name)
	}

	// Call the tool
	result, err := toolToCall.WrappedFunction(ctx, []byte(tc.Arguments))
	if err != nil {
		return nil, fmt.Errorf("tool execution failed: %w", err)
	}

	// Add the tool response to messages
	return &message.Message{
		Role:    message.ToolMessageRole,
		Content: fmt.Sprintf("%v", result),
	}, nil
}

// AddTool adds a tool to the agent's available tools
func (a *DefaultAgent) AddTool(tool tool.Tool) error {
	if tool.Name == "" {
		return errors.New("tool must have a name")
	}

	if tool.WrappedFunction == nil {
		return errors.New("tool must have a function")
	}

	a.tools[tool.Name] = tool

	return nil
}

// Example stop condition
func DefaultStopCondition(step *agent.AgentStep) bool {
	// Stop if there's an error
	if step.Error != nil {
		return true
	}

	// Stop if no tool calls were made and we got a response
	if len(step.Message.ToolCalls) == 0 && len(step.Message.Content) != 0 {
		return true
	}

	return false
}
