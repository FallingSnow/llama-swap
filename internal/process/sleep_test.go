package process

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"reflect"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/logmon"
)

// TestSleeper_CudaCheckpointArgs checks the exact argv passed to the
// cuda-checkpoint binary for each action.
func TestSleeper_CudaCheckpointArgs(t *testing.T) {
	tests := []struct {
		action string
		pid    int
		want   []string
	}{
		{"checkpoint", 4242, []string{"--action", "checkpoint", "--pid", "4242"}},
		{"restore", 7, []string{"--action", "restore", "--pid", "7"}},
	}
	for _, tt := range tests {
		got := cudaCheckpointArgs(tt.action, tt.pid)
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("cudaCheckpointArgs(%q, %d) = %v, want %v", tt.action, tt.pid, got, tt.want)
		}
	}
}

// TestSleeper_New selects the right sleeper for each configured swapOutStrategy.
func TestSleeper_New(t *testing.T) {
	logger := logmon.NewWriter(io.Discard)

	tests := []struct {
		strategy string
		wantNil  bool
	}{
		{"", true}, // default: no sleeper
		// cuda-checkpoint yields a sleeper on Linux, falls back to nil elsewhere.
		{"cuda-checkpoint", runtime.GOOS != "linux"},
		{"not-a-real-strategy", true}, // unknown: fallback
	}
	for _, tt := range tests {
		got := newSleeper(config.ModelConfig{SwapOutStrategy: tt.strategy}, logger)
		if (got == nil) != tt.wantNil {
			t.Errorf("newSleeper(%q) nil = %v, want nil = %v (GOOS=%s)", tt.strategy, got == nil, tt.wantNil, runtime.GOOS)
		}
	}
}

// TestSleeper_CudaCheckpointNilProcess returns an error when the process has no
// pid, rather than panicking.
func TestSleeper_CudaCheckpointNilProcess(t *testing.T) {
	logger := logmon.NewWriter(io.Discard)
	s := &cudaCheckpointSleeper{log: logger}
	ctx := context.Background()
	cmd := &exec.Cmd{} // Process is nil
	if err := s.Suspend(ctx, cmd); err == nil {
		t.Error("Suspend(nil process) = nil, want error")
	}
	if err := s.Resume(ctx, cmd); err == nil {
		t.Error("Resume(nil process) = nil, want error")
	}
}

// fakeSleeper records Suspend/Resume calls and can be told to fail, so the
// state-machine tests can exercise suspend/restore without a GPU or the
// cuda-checkpoint binary.
type fakeSleeper struct {
	suspendErr error
	resumeErr  error
	// suspend and resume are incremented from the process run-loop goroutine
	// (inside Suspend/Resume) and read from the test's main goroutine, so they
	// are atomic to stay clean under go test -race.
	suspend atomic.Int32
	resume  atomic.Int32
	// suspendFails, when set, makes Suspend return an error so a Stop falls
	// through to a real kill. Tests set it in cleanup to ensure a suspended
	// process is terminated rather than left alive.
	suspendFails atomic.Bool
	// lastCmd captures the process being suspended so a test can kill it to
	// simulate death while suspended. Guarded by mu; it is written by Suspend
	// (from the run-loop goroutine or the test goroutine during cleanup) and
	// read by the test's main goroutine.
	mu      sync.Mutex
	lastCmd *exec.Cmd
}

func (f *fakeSleeper) Suspend(_ context.Context, cmd *exec.Cmd) error {
	f.suspend.Add(1)
	if cmd != nil {
		f.mu.Lock()
		f.lastCmd = cmd
		f.mu.Unlock()
	}
	if f.suspendFails.Load() {
		return errors.New("suspend disabled for cleanup")
	}
	return f.suspendErr
}

// capturedCmd returns the last suspended process, if any.
func (f *fakeSleeper) capturedCmd() *exec.Cmd {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.lastCmd
}

func (f *fakeSleeper) Resume(_ context.Context, cmd *exec.Cmd) error {
	f.resume.Add(1)
	return f.resumeErr
}

// TestProcessCommand_SuspendRestore covers the happy path: a model that is
// offloaded via a sleeper moves ready→suspended on Stop (the process stays
// alive) and suspended→ready on EnsureReady (restored in place, not restarted).
func TestProcessCommand_SuspendRestore(t *testing.T) {
	skipIfNoSimpleResponder(t)

	cmd, port := simpleResponderCmd(t, "-silent")
	p := newProcessCommand(t, config.ModelConfig{
		Cmd:                cmd,
		Proxy:              fmt.Sprintf("http://127.0.0.1:%d", port),
		CheckEndpoint:      "/health",
		HealthCheckTimeout: 10,
	})
	fake := &fakeSleeper{}
	p.sleeper = fake
	t.Cleanup(func() {
		fake.suspendFails.Store(true)
		if err := p.Stop(testStopTimeout); err != nil {
			t.Errorf("cleanup Stop: %v", err)
		}
	})

	// Bring it up and confirm it is ready.
	ch := runAsync(t, p)
	defer func() { <-ch }()

	// Unload it: with a sleeper this suspends rather than kills.
	if err := p.Stop(testStopTimeout); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	waitForState(t, p, StateSuspended)
	if got := fake.suspend.Load(); got != 1 {
		t.Errorf("Suspend called %d times, want 1", got)
	}

	// A suspended process is not serving: ServeHTTP must return 503.
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	p.ServeHTTP(rr, req)
	if rr.Code != http.StatusServiceUnavailable {
		t.Errorf("suspended ServeHTTP status = %d, want %d", rr.Code, http.StatusServiceUnavailable)
	}

	// Swap it back in: EnsureReady restores it in place.
	ctx, cancel := context.WithTimeout(context.Background(), testStartTimeout)
	defer cancel()
	if err := p.EnsureReady(ctx, testStartTimeout); err != nil {
		t.Fatalf("EnsureReady after suspend: %v", err)
	}
	waitForState(t, p, StateReady)
	if got := fake.resume.Load(); got != 1 {
		t.Errorf("Resume called %d times, want 1", got)
	}
}

// TestProcessCommand_SuspendFallsBackToKill covers the fallback: when Suspend
// fails, Stop must kill the process and leave it Stopped (not suspended).
func TestProcessCommand_SuspendFallsBackToKill(t *testing.T) {
	skipIfNoSimpleResponder(t)

	cmd, port := simpleResponderCmd(t, "-silent")
	p := newProcessCommand(t, config.ModelConfig{
		Cmd:                cmd,
		Proxy:              fmt.Sprintf("http://127.0.0.1:%d", port),
		CheckEndpoint:      "/health",
		HealthCheckTimeout: 10,
	})
	fake := &fakeSleeper{suspendErr: errors.New("suspend failed")}
	p.sleeper = fake
	t.Cleanup(func() {
		fake.suspendFails.Store(true)
		if err := p.Stop(testStopTimeout); err != nil {
			t.Errorf("cleanup Stop: %v", err)
		}
	})

	ch := runAsync(t, p)
	defer func() { <-ch }()

	if err := p.Stop(testStopTimeout); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	// Suspend failed, so the process should be killed and Stopped.
	waitForState(t, p, StateStopped)
	if got := fake.suspend.Load(); got != 1 {
		t.Errorf("Suspend called %d times, want 1", got)
	}
	// It must be restartable as a fresh process afterwards.
	ctx, cancel := context.WithTimeout(context.Background(), testStartTimeout)
	defer cancel()
	if err := p.EnsureReady(ctx, testStartTimeout); err != nil {
		t.Fatalf("EnsureReady after failed suspend: %v", err)
	}
	waitForState(t, p, StateReady)
}

// TestProcessCommand_DiedWhileSuspended covers the edge where a suspended
// process exits on its own (cmdDone closes). The run loop must transition out
// of StateSuspended to StateStopped; it must not stay stuck in StateSuspended
// (which would make a subsequent Run try to restore a dead process).
func TestProcessCommand_DiedWhileSuspended(t *testing.T) {
	skipIfNoSimpleResponder(t)

	cmd, port := simpleResponderCmd(t, "-silent")
	p := newProcessCommand(t, config.ModelConfig{
		Cmd:                cmd,
		Proxy:              fmt.Sprintf("http://127.0.0.1:%d", port),
		CheckEndpoint:      "/health",
		HealthCheckTimeout: 10,
	})
	fake := &fakeSleeper{}
	p.sleeper = fake
	t.Cleanup(func() {
		fake.suspendFails.Store(true)
		if err := p.Stop(testStopTimeout); err != nil {
			t.Errorf("cleanup Stop: %v", err)
		}
	})

	// Bring it up and confirm it is ready.
	ch := runAsync(t, p)
	defer func() { <-ch }()

	// Unload it: with a sleeper this suspends rather than kills.
	if err := p.Stop(testStopTimeout); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	waitForState(t, p, StateSuspended)

	// Simulate the process dying while suspended.
	suspended := fake.capturedCmd()
	if suspended == nil || suspended.Process == nil {
		t.Fatalf("suspended process cmd not captured")
	}
	if err := suspended.Process.Kill(); err != nil {
		t.Fatalf("kill suspended process: %v", err)
	}

	// The run loop must notice the exit and drop out of StateSuspended.
	waitForState(t, p, StateStopped)
}
