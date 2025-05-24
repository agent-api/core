package bootstrap

import (
	"github.com/go-logr/logr"

	"github.com/agent-api/core"
)

// NewAgentConfig holds configuration for agent initialization
type NewAgentConfig struct {
	// The core.Provider the agent will use
	Provider core.Provider

	// Maximum number of steps before forcing stop
	MaxSteps int

	// Tools the agent has access to execute with
	Tools []*core.Tool

	// VecStore is the vector store configured for the agent
	VecStore core.VectorStorer

	// System prompt
	SystemPrompt string

	// The provided logr.Logger
	Logger *logr.Logger

	// The configured memory backend that stores messages across agent runs.
	Memory core.MemoryBackend

	// Maximum number of messages restored from memory
	// default 10
	MaxMemoryWindowContext int
}

// RunOptionFunc is a function type that modifies RunOptions
type NewAgentConfigFunc func(*NewAgentConfig)

func WithProvider(provider core.Provider) NewAgentConfigFunc {
	return func(conf *NewAgentConfig) {
		conf.Provider = provider
	}
}

func WithMaxSteps(steps int) NewAgentConfigFunc {
	return func(conf *NewAgentConfig) {
		conf.MaxSteps = steps
	}
}

func WithTools(tool ...*core.Tool) NewAgentConfigFunc {
	return func(conf *NewAgentConfig) {
		if conf.Tools == nil {
			conf.Tools = []*core.Tool{}
		}

		conf.Tools = append(conf.Tools, tool...)
	}
}

func WithVectorStore(vecStore core.VectorStorer) NewAgentConfigFunc {
	return func(conf *NewAgentConfig) {
		conf.VecStore = vecStore
	}
}

func WithSystemPrompt(prompt string) NewAgentConfigFunc {
	return func(conf *NewAgentConfig) {
		conf.SystemPrompt = prompt
	}
}

func WithLogger(l *logr.Logger) NewAgentConfigFunc {
	return func(conf *NewAgentConfig) {
		conf.Logger = l
	}
}

func WithMemory(m core.MemoryBackend) NewAgentConfigFunc {
	return func(conf *NewAgentConfig) {
		conf.Memory = m
	}
}

func WithMaxMemoryWindowContext(size int) NewAgentConfigFunc {
	return func(conf *NewAgentConfig) {
		conf.MaxMemoryWindowContext = size
	}
}
