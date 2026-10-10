package process

import (
	"context"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/logmon"
)

// Sleeper is a strategy for offloading a running model's accelerator state
// without killing the process, and restoring it later. Suspend frees the GPU
// (or other accelerator) while the process stays alive; Resume re-acquires it.
// Neither performs a health check — the caller handles readiness, so a sleeper
// that fails can be transparently replaced by a normal kill/restart.
type Sleeper interface {
	// Suspend offloads the accelerator state of the process led by cmd.
	// Returns an error if the state could not be suspended. The process may
	// be left in a partially suspended state (e.g. locked but not yet
	// checkpointed), in which case the caller should fall back to a kill.
	// The context bounds the operation so a stuck driver call can be escaped
	// by the caller; each individual step is additionally capped by
	// sleepTimeout.
	Suspend(ctx context.Context, cmd *exec.Cmd) error
	// Resume re-acquires the accelerator state previously offloaded by
	// Suspend for the process led by cmd. The context bounds the operation as
	// in Suspend.
	Resume(ctx context.Context, cmd *exec.Cmd) error
}

// sleepTimeout bounds a single cuda-checkpoint Suspend/Resume invocation.
// Copying a large model's GPU memory to/from the host can take a while, so this
// is generous; it is a safety net against a hung driver call, not a target.
const sleepTimeout = 120 * time.Second

// newSleeper returns the Sleeper selected by conf.SwapOutStrategy, or nil when
// no sleeper is configured (the default kill/restart behavior). On unsupported
// platforms a warning is logged and nil is returned so the caller falls back to
// the normal path.
func newSleeper(conf config.ModelConfig, log *logmon.Monitor) Sleeper {
	switch conf.SwapOutStrategy {
	case "":
		return nil
	case config.SwapOutStrategyCudaCheckpoint:
		if runtime.GOOS != "linux" {
			log.Warnf("<%s> swapOutStrategy 'cuda-checkpoint' requires Linux; falling back to kill/restart", conf.Name)
			return nil
		}
		return &cudaCheckpointSleeper{log: log}
	default:
		log.Warnf("<%s> unknown swapOutStrategy %q; falling back to kill/restart", conf.Name, conf.SwapOutStrategy)
		return nil
	}
}

// cudaCheckpointSleeper offloads a process's CUDA state using NVIDIA's
// cuda-checkpoint utility. Suspend runs `cuda-checkpoint --action checkpoint
// --pid <pid>`, which copies device memory to the host and releases the GPU
// while the process keeps running; Resume runs `--action restore --pid <pid>`
// to re-acquire the GPU and copy the memory back.
//
// log is the owning process's processLogger. The helper only knows the pid (not
// the model id), so its own log lines key on the pid; the surrounding
// process-level log lines carry the model id prefix.
type cudaCheckpointSleeper struct {
	log *logmon.Monitor
}

var _ Sleeper = (*cudaCheckpointSleeper)(nil)

// cudaCheckpointArgs builds the argument list for a cuda-checkpoint invocation
// with the given action against pid. Extracted for testing.
func cudaCheckpointArgs(action string, pid int) []string {
	return []string{"--action", action, "--pid", fmt.Sprintf("%d", pid)}
}

// checkpointActions is the sequence of cuda-checkpoint actions to suspend a
// process. Lock first to prevent new CUDA allocations (this is what makes the
// state compatible with checkpoint), then checkpoint to copy GPU memory to
// the host.
var checkpointActions = []string{"lock", "checkpoint"}

// restoreActions is the sequence to resume a suspended process. Restore first
// to copy memory back to the GPU, then unlock to allow new CUDA allocations.
var restoreActions = []string{"restore", "unlock"}

// runAction invokes cuda-checkpoint once with the given action against pid,
// bounded by sleepTimeout. The timeout cancel is deferred so the timer is
// released even if Run panics.
func (s *cudaCheckpointSleeper) runAction(ctx context.Context, action string, pid int) error {
	args := cudaCheckpointArgs(action, pid)
	s.log.Infof("cuda-checkpoint %s for pid %d: %s", action, pid, strings.Join(args, " "))

	stepStart := time.Now()
	runCtx, cancel := context.WithTimeout(ctx, sleepTimeout)
	defer cancel()

	c := exec.CommandContext(runCtx, "cuda-checkpoint", args...)
	c.Stdout = s.log
	c.Stderr = s.log
	c.WaitDelay = 10 * time.Second
	c.Cancel = func() error {
		// cuda-checkpoint is a short-lived helper, not a long-lived
		// upstream, so a plain kill of its (single) process is sufficient;
		// there is no process group to reap.
		if c.Process != nil {
			return c.Process.Kill()
		}
		return nil
	}

	err := c.Run()
	if err != nil {
		elapsed := time.Since(stepStart)
		if runCtx.Err() == context.DeadlineExceeded {
			s.log.Warnf("cuda-checkpoint %s for pid %d timed out after %v", action, pid, elapsed)
			return fmt.Errorf("cuda-checkpoint %s timed out after %v: %w", action, sleepTimeout, runCtx.Err())
		}
		if runCtx.Err() == context.Canceled {
			s.log.Warnf("cuda-checkpoint %s for pid %d cancelled after %v", action, pid, elapsed)
			return fmt.Errorf("cuda-checkpoint %s cancelled: %w", action, runCtx.Err())
		}
		s.log.Warnf("cuda-checkpoint %s for pid %d failed after %v: %v", action, pid, elapsed, err)
		return fmt.Errorf("cuda-checkpoint %s failed: %w", action, err)
	}
	s.log.Debugf("cuda-checkpoint %s for pid %d completed in %v", action, pid, time.Since(stepStart))
	return nil
}

func (s *cudaCheckpointSleeper) run(ctx context.Context, actions []string, cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil {
		return fmt.Errorf("cuda-checkpoint: no process to %s", strings.Join(actions, "+"))
	}
	pid := cmd.Process.Pid

	totalStart := time.Now()
	for _, action := range actions {
		if err := s.runAction(ctx, action, pid); err != nil {
			return err
		}
	}
	s.log.Infof("cuda-checkpoint [%s] completed for pid %d in %v", strings.Join(actions, " → "), pid, time.Since(totalStart))
	return nil
}

func (s *cudaCheckpointSleeper) Suspend(ctx context.Context, cmd *exec.Cmd) error {
	return s.run(ctx, checkpointActions, cmd)
}

func (s *cudaCheckpointSleeper) Resume(ctx context.Context, cmd *exec.Cmd) error {
	return s.run(ctx, restoreActions, cmd)
}
