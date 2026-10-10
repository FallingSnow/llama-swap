package process

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os/exec"
	"strings"
	"sync/atomic"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/event"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

var ErrStartAborted = fmt.Errorf("aborted")

// errPrematureExit signals that the upstream process exited before becoming
// ready. pollHealth returns it so a caller can distinguish a premature exit
// (just cancel the ctx, report "exited prematurely") from an abort (kill the
// process, report the original error).
var errPrematureExit = fmt.Errorf("upstream command exited prematurely")

// healthCheckKey marks requests issued by the health check loop, which polls
// the upstream through the same reverse proxy. Their failures are the expected
// shape of a model still booting, so they must not be logged as proxy errors.
type healthCheckKey struct{}

// newProxyErrorHandler builds the ErrorHandler for a model's reverse proxy.
//
// httputil.ReverseProxy's default handler answers every failure with 502, so a
// client hanging up mid-generation is logged as a Bad Gateway and sends
// operators looking at an inference server that was healthy the whole time.
// Cancellation is classified here, where the error is actually known: the
// request is recorded with the client-closed sentinel rather than blamed on the
// upstream, and at debug level because an impatient caller is normal traffic.
// Real upstream failures keep the 502. See #1029.
func newProxyErrorHandler(id string, proxyLogger *logmon.Monitor) func(http.ResponseWriter, *http.Request, error) {
	return func(w http.ResponseWriter, r *http.Request, err error) {
		// Pick the log level first: a cancelled request is never an upstream
		// fault, whether or not the sentinel ends up applying below.
		switch {
		case errors.Is(err, context.Canceled) || r.Context().Err() != nil:
			proxyLogger.Debugf("<%s> request cancelled: %v", id, err)
		case r.Context().Value(healthCheckKey{}) != nil:
			proxyLogger.Debugf("<%s> health check not ready: %v", id, err)
		default:
			proxyLogger.Warnf("<%s> proxy error: %v", id, err)
		}

		// Only a client that actually hung up gets the recorded-only sentinel.
		// A request cancelled server-side (an operator cancelling it from the
		// UI, or shutdown) still has a client waiting, and must be answered —
		// otherwise net/http finalizes it as an empty 200, telling the caller
		// the request succeeded.
		if swaputil.MarkClientClosed(w, r) || swaputil.ResponseStarted(w) {
			return
		}
		w.WriteHeader(http.StatusBadGateway)
	}
}

// cmdWaitDelay is the upper bound the runtime will wait for child I/O to
// drain after the process exits before force-closing the stdout/stderr
// pipes. Required so that cmd.Wait() returns even when a forked grandchild
// inherits and holds the pipes open (e.g. a shell wrapper that backgrounds
// the real binary). killProcess sends the stop signal directly (not via the
// cmd context), so this delay is measured from process exit rather than from
// the stop request, and stays independent of the caller's graceful timeout.
const cmdWaitDelay = 10 * time.Second

// parentCancelGraceTimeout is the graceful timeout used when the process is
// torn down because parentCtx was cancelled (final router teardown or app
// shutdown). In the normal flow the process has already been stopped via
// Stop() by this point, so killProcess is a no-op kill; the short grace just
// bounds the rare case where a process is still alive when its context is cut.
const parentCancelGraceTimeout = time.Second

// startReq asks the run loop to bring the process up. Run and EnsureReady share
// this one request type — and therefore one code path — so there is only ever a
// single way to start a process. block selects the caller's semantics: Run parks
// its response until the process terminates, EnsureReady is answered as soon as
// the process is ready or the start fails.
type startReq struct {
	timeout time.Duration
	respond chan error
	block   bool
}

type stopReq struct {
	timeout time.Duration
	force   bool
	respond chan error
}

type waitReadyReq struct {
	respond chan error
}

type startResult struct {
	cmd       *exec.Cmd
	cmdDone   <-chan struct{}
	cancel    context.CancelFunc
	handlerFn http.HandlerFunc
	err       error
}

type ProcessCommand struct {
	id        string
	config    config.ModelConfig
	parentCtx context.Context

	processLogger *logmon.Monitor
	proxyLogger   *logmon.Monitor

	// sleeper, when non-nil, offloads the model's accelerator state on
	// unload (freeing the GPU while the process stays alive) and restores it
	// on swap-in, instead of killing and restarting the process. nil means
	// the default kill/restart behavior.
	sleeper Sleeper

	// waitDelay is assigned to cmd.WaitDelay when starting the upstream
	// process. Defaults to cmdWaitDelay; tests override it to keep the
	// pipe-close backstop from dominating their runtime.
	waitDelay time.Duration

	startCh     chan startReq
	stopCh      chan stopReq
	waitReadyCh chan waitReadyReq

	// current Status. Written only by run(); read by State() and Status()
	// via atomic load.
	status atomic.Pointer[Status]

	// stores the active reverse-proxy handler when the process is running.
	// Written only by run(); read by ServeHTTP via atomic load.
	handler atomic.Pointer[http.HandlerFunc]

	// lastUse is the unix-nano timestamp of the most recent activity baseline.
	// It is initialized when the process becomes Ready and updated after ServeHTTP completes.
	lastUse  atomic.Int64
	inflight atomic.Int64 // current in-flight ServeHTTP calls
}

var _ Process = (*ProcessCommand)(nil)

func New(
	parentCtx context.Context,
	id string,
	conf config.ModelConfig,
	processLogger *logmon.Monitor,
	proxyLogger *logmon.Monitor,
) (*ProcessCommand, error) {
	p := &ProcessCommand{
		id:            id,
		config:        conf,
		parentCtx:     parentCtx,
		processLogger: processLogger,
		proxyLogger:   proxyLogger, sleeper: newSleeper(conf, processLogger),
		startCh:     make(chan startReq),
		stopCh:      make(chan stopReq),
		waitReadyCh: make(chan waitReadyReq),
		waitDelay:   cmdWaitDelay,
	}
	p.status.Store(&Status{State: StateStopped})

	go p.run()
	return p, nil
}

func (p *ProcessCommand) Logger() *logmon.Monitor { return p.processLogger }

// run is the single-writer goroutine that owns all mutable lifecycle state
// (current ProcessState, the running *exec.Cmd, the active reverse-proxy
// handler, and the list of WaitReady subscribers). Every public method
// (Run / Stop / State / WaitReady) is a thin client that sends a request on
// one of the channels below and waits for a response — this funnels concurrent
// callers through a single serialization point so the state machine never
// observes a race.
func (p *ProcessCommand) run() {
	// Mutable state — only read/written from this goroutine. ServeHTTP reads
	// p.handler concurrently, which is why handler is an atomic.Pointer.
	// p.status mirrors `state` so State() can observe transitions; setState
	// writes both.
	state := StateStopped
	setState := func(s ProcessState) {
		old := state
		state = s
		next := &Status{State: s}
		if s == StateReady {
			if old == StateReady {
				next.ReadySince = p.status.Load().ReadySince
			} else {
				next.ReadySince = time.Now()
			}
		}
		p.status.Store(next)
		if old != s {
			event.Emit(swaputil.ProcessStateChangeEvent{
				ProcessName: p.id,
				OldState:    string(old),
				NewState:    string(s),
			})
		}
	}
	var (
		cmd          *exec.Cmd
		cmdDone      <-chan struct{}
		cmdCancel    context.CancelFunc
		readyWaiters []waitReadyReq
		// runResp parks the in-flight Run caller's response channel. The
		// interface contract is that Run blocks until the process is
		// terminated, so we hold this until Stop, parentCtx, or an
		// upstream exit unblocks it via respondRun.
		runResp chan<- error
	)

	// notifyWaiters wakes every blocked WaitReady caller with the given result.
	// It must be called on every transition into a state that resolves the
	// "is it ready yet?" question — ready, failed, aborted, shutdown, stopped,
	// or exited. Missing one strands the subscriber forever (issue #946).
	notifyWaiters := func(err error) {
		for _, w := range readyWaiters {
			select {
			case w.respond <- err:
			default:
			}
		}
		readyWaiters = nil
	}

	// respondRun delivers the final Run result, if a Run caller is parked.
	respondRun := func(err error) {
		if runResp != nil {
			runResp <- err
			runResp = nil
		}
	}

	for {
		select {
		// Shutdown: parent context cancelled. Tear down any running process,
		// wake any pending WaitReady callers with an error, then exit the
		// goroutine permanently. Subsequent public-method calls will fail
		// because parentCtx.Done() unblocks their send-side selects.
		case <-p.parentCtx.Done():
			// Mark shutdown before killProcess so concurrent State() readers
			// stop treating this process as ready while the (possibly slow)
			// teardown is in progress.
			setState(StateShutdown)
			if cmd != nil {
				p.handler.Store(nil)
				p.killProcess(cmd, cmdCancel, cmdDone, parentCancelGraceTimeout)
				cmd = nil
				cmdDone = nil
				cmdCancel = nil
			}
			notifyWaiters(fmt.Errorf("[%s] shutdown", p.id))
			respondRun(fmt.Errorf("[%s] shutdown", p.id))
			return

		// Upstream exited on its own (not via Stop). Drop handler state,
		// transition to Stopped, and unblock the parked Run caller.
		// cmdDone is nil while no process is running, so this case is dormant
		// in StateStopped. It also fires while StateSuspended: the process
		// stays alive (and cmdDone latched) after a suspend, so a process that
		// dies while offloaded reaches StateStopped here rather than being
		// left stuck in StateSuspended for a later Run to try to restore.
		case <-cmdDone:
			if cmdCancel != nil {
				cmdCancel()
			}
			cmd = nil
			cmdDone = nil
			cmdCancel = nil
			p.handler.Store(nil)
			setState(StateStopped)
			p.proxyLogger.Warnf("<%s> upstream process exited unexpectedly", p.id)
			// Safety net: readyWaiters is normally empty here because
			// WaitReady is answered immediately while StateReady. Notifying
			// anyway keeps the invariant that no transition into a settled
			// state can leave a subscriber parked forever.
			notifyWaiters(fmt.Errorf("[%s] upstream exited unexpectedly", p.id))
			respondRun(fmt.Errorf("[%s] upstream exited unexpectedly", p.id))

		// WaitReady: if we're already in a terminal-for-this-question state,
		// respond immediately; otherwise queue the caller and let a future
		// state transition wake them via notifyWaiters.
		case req := <-p.waitReadyCh:
			switch state {
			case StateReady:
				req.respond <- nil
			case StateShutdown:
				req.respond <- fmt.Errorf("[%s] shutdown", p.id)
			default:
				readyWaiters = append(readyWaiters, req)
			}

		// Start the upstream process (Run or EnsureReady — see startReq).
		// doStart can take a long time (health-check polling), so it runs in
		// a separate goroutine and we wait on resultCh. While waiting we also
		// listen for an incoming Stop — that's how callers cancel an in-flight
		// start.
		case req := <-p.startCh:
			// EnsureReady answers straight from the current state when the
			// "is it ready?" question is already settled. There is deliberately
			// no StateStopping case: a stop keeps this loop parked inside
			// killProcess and out of the select, so a request can only ever be
			// received once the stop has finished and state is Stopped again.
			// Waiting on the channel IS the synchronisation — the caller never
			// has to guess the state from outside.
			if !req.block {
				switch state {
				case StateReady:
					req.respond <- nil
					continue
				case StateShutdown:
					req.respond <- fmt.Errorf("[%s] shutdown", p.id)
					continue
				}
			}
			// Only valid from StateStopped or StateSuspended. For Run this is
			// also the "second Run while already running" rejection.
			if state != StateStopped && state != StateSuspended {
				req.respond <- fmt.Errorf("[%s] could not be started in %s state", p.id, state)
				continue
			}
			// wasSuspended captures whether we are restoring an offloaded
			// process (StateSuspended) versus starting a fresh one (Stopped).
			// It is read inside the start goroutine below, so capture it now
			// before setState mutates the shared state variable.
			wasSuspended := state == StateSuspended
			setState(StateStarting)

			startCtx, cancelStart := context.WithCancel(context.Background())
			resultCh := make(chan startResult, 1)
			go func() {
				if wasSuspended {
					// Restore the offloaded process in place. If the restore
					// fails (sleeper error, health-check timeout, or the
					// process died while suspended) the process is no longer
					// usable, so tear it down and fall back to a fresh start.
					res := p.doRestore(startCtx, req.timeout, cmd, cmdDone, cmdCancel)
					if res.err != nil {
						p.killProcess(cmd, cmdCancel, cmdDone, 5*time.Second)
						res = p.doStart(startCtx, req.timeout)
					}
					resultCh <- res
					return
				}
				resultCh <- p.doStart(startCtx, req.timeout)
			}()

			// pendingStop holds a Stop request that arrived mid-start, so we
			// can respond to it AFTER we've finished tearing the start down.
			var pendingStop *stopReq
			select {
			// doStart finished on its own — either successfully (latch
			// cmd/handler and move to Ready) or with an error (back to
			// Stopped). Either way wake WaitReady subscribers and reply
			// to the Run caller.
			case res := <-resultCh:
				if res.err == nil {
					cmd = res.cmd
					cmdDone = res.cmdDone
					cmdCancel = res.cancel
					fn := res.handlerFn
					p.handler.Store(&fn)
					// A newly ready process starts a fresh idle window. Without this,
					// lastUse is zero on first start or stale after a restart, so TTL
					// can unload it on the first one-second ticker tick.
					p.lastUse.Store(time.Now().UnixNano())
					setState(StateReady)
					notifyWaiters(nil)
					if req.block {
						// Park the Run response — Run blocks until the process
						// terminates, so we only fire this when Stop, parentCtx,
						// or the upstream exit takes the process down.
						runResp = req.respond
					} else {
						// EnsureReady's question is answered: it's ready.
						req.respond <- nil
					}

					// Start TTL goroutine if configured — self-terminates
					// when state leaves StateReady.
					if p.config.UnloadAfter > 0 {
						ttlDuration := time.Duration(p.config.UnloadAfter) * time.Second
						go func() {
							ticker := time.NewTicker(time.Second)
							defer ticker.Stop()
							for range ticker.C {
								if p.State() != StateReady {
									return
								}
								if p.inflight.Load() != 0 {
									continue
								}
								if time.Since(time.Unix(0, p.lastUse.Load())) > ttlDuration {
									p.proxyLogger.Infof("<%s> Unloading model, TTL of %ds reached", p.id, p.config.UnloadAfter)
									p.Stop(time.Duration(p.config.UnloadTimeout) * time.Second)
									return
								}
							}
						}()
					}
				} else {
					setState(StateStopped)
					notifyWaiters(res.err)
					req.respond <- res.err
				}

			// Stop arrived while doStart was still running. Cancel the
			// start context to abort it, then wait for doStart to return.
			// If doStart had already crossed the finish line before
			// cancellation took effect, it returns a live cmd that we
			// must kill ourselves. The Run caller gets ErrAbort; the Stop
			// caller is parked in pendingStop and answered below.
			case stop := <-p.stopCh:
				cancelStart()
				res := <-resultCh
				if res.cmd != nil {
					p.killProcess(res.cmd, res.cancel, res.cmdDone, stop.timeout)
				}
				setState(StateStopped)
				notifyWaiters(ErrStartAborted)
				req.respond <- ErrStartAborted
				pendingStop = &stop

			// Parent context cancelled (e.g. config reload) while doStart
			// was still running. Stop() returns early when parentCtx is
			// done and never sends on stopCh, so we must handle shutdown
			// here to avoid leaving doStart running indefinitely.
			case <-p.parentCtx.Done():
				cancelStart()
				// Mark shutdown before tearing the process down: killProcess
				// may block (e.g. taskkill on Windows is slow to spawn), and
				// callers observing State() should see StateShutdown promptly
				// rather than a stale StateStarting.
				setState(StateShutdown)
				res := <-resultCh
				if res.cmd != nil {
					p.killProcess(res.cmd, res.cancel, res.cmdDone, parentCancelGraceTimeout)
				}
				notifyWaiters(fmt.Errorf("[%s] shutdown", p.id))
				respondRun(fmt.Errorf("[%s] shutdown", p.id))
				return
			}
			// cancelStart is idempotent; calling it again here ensures the
			// context is released even on the success path (govet leak check).
			cancelStart()
			if pendingStop != nil {
				pendingStop.respond <- nil
			}

		// Stop: tear down a running process. When a sleeper is configured and
		// the process is ready, we suspend it (offloading the accelerator state
		// and freeing the GPU while the process stays alive) instead of
		// killing it. A Stop that arrives while a process is already suspended
		// is a no-op.
		case stop := <-p.stopCh:
			// Already suspended: the GPU is free and the process is alive.
			// A second Stop is a no-op — killing here would destroy the
			// offloaded state and force a full reload on next use. A forced
			// Stop skips this guard and kills the suspended process.
			if state == StateSuspended && !stop.force {
				stop.respond <- nil
				continue
			}
			// A sleeper can only suspend a process that is ready to serve.
			// Suspending from Starting/Stopped/Shutdown is meaningless, so it
			// falls through to a normal kill. During parent shutdown the
			// process will be killed shortly anyway, so skip the (slow)
			// suspend and go straight to kill. A forced Stop skips the
			// sleeper entirely.
			if !stop.force && p.sleeper != nil && cmd != nil && state == StateReady && p.parentCtx.Err() == nil {
				setState(StateStopping)
				// Bound the suspend to parentCtx so a shutdown that starts while
				// the checkpoint is in flight can escape it rather than waiting
				// out the full sleepTimeout.
				if err := p.suspendProcess(p.parentCtx, cmd); err == nil {
					// Suspended: the process stays alive (cmd stays latched) but
					// no longer holds a GPU and is not serving, so drop the
					// handler and move to the suspended state. cmdCancel is left
					// set — it is the safety net for this live process and is
					// released when the process is later killed or restored.
					p.handler.Store(nil)
					setState(StateSuspended)
					notifyWaiters(fmt.Errorf("[%s] suspended", p.id))
					respondRun(fmt.Errorf("[%s] suspended", p.id))
					stop.respond <- nil
					continue
				}
				// Suspend failed: fall through to a normal kill below.
			}
			// Falling through to a kill covers: no sleeper, suspend just
			// failed, or a Stop from Starting/Stopped states.
			toreDown := cmd != nil
			if cmd != nil {
				setState(StateStopping)
				p.killProcess(cmd, cmdCancel, cmdDone, stop.timeout)
				cmd = nil
				cmdDone = nil
				cmdCancel = nil
				p.handler.Store(nil)
			}
			// Stop is a no-op (and not an error) when already Stopped — this
			// is what makes it idempotent for callers that don't track state.
			setState(StateStopped)
			if toreDown {
				// A process we just killed is not going to become ready, so
				// release anyone parked on that question rather than leaving
				// them stranded (issue #946). Guarded on having actually torn
				// something down: an idempotent no-op Stop must not cancel a
				// subscriber waiting on a start that hasn't happened yet, which
				// is the legitimate `go Run(); WaitReady()` ordering.
				notifyWaiters(fmt.Errorf("[%s] stopped", p.id))
			}
			respondRun(nil)
			stop.respond <- nil
		}
	}
}

func (p *ProcessCommand) doStart(startCtx context.Context, healthCheckTimeout time.Duration) startResult {
	if p.config.Proxy == "" {
		return startResult{err: fmt.Errorf("upstream proxy missing")}
	}

	args, err := p.config.SanitizedCommand()
	if err != nil {
		return startResult{err: fmt.Errorf("unable to get sanitized command: %w", err)}
	}

	if _, err := url.Parse(p.config.Proxy); err != nil {
		return startResult{err: fmt.Errorf("invalid proxy URL %q: %w", p.config.Proxy, err)}
	}

	handlerFn, reverseProxy := p.buildHandler()

	// cmdCtx + cmd.Cancel are wired as a safety net: if the context is ever
	// cancelled while the process is alive, cmd.Cancel sends SIGTERM / CmdStop
	// and the runtime escalates to SIGKILL after cmd.WaitDelay. In the normal
	// teardown path killProcess sends the stop signal directly instead, so
	// cmd.WaitDelay only acts as the inherited-pipe backstop measured from
	// process exit (see killProcess).
	cmdCtx, cmdCancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(cmdCtx, args[0], args[1:]...)
	cmd.Stderr = p.processLogger
	cmd.Stdout = p.processLogger
	cmd.Env = append(cmd.Environ(), p.config.Env...)
	cmd.Cancel = func() error { return p.sendStopSignal(cmd) }
	cmd.WaitDelay = p.waitDelay
	setProcAttributes(cmd)

	p.proxyLogger.Debugf("<%s> Executing start command: %s, env: %s", p.id, strings.Join(args, " "), strings.Join(p.config.Env, ", "))

	cmdDone := make(chan struct{})
	if err := cmd.Start(); err != nil {
		cmdCancel()
		return startResult{err: fmt.Errorf("failed to start command '%s': %w", strings.Join(args, " "), err)}
	}

	go func() {
		waitErr := cmd.Wait()
		switch st := p.State(); {
		case waitErr == nil:
			p.proxyLogger.Debugf("<%s> process exited cleanly", p.id)
		case st == StateStopping || st == StateShutdown:
			// Expected: we force-terminated the process. A forced kill exits
			// the child with a non-zero code (e.g. taskkill /f on Windows
			// yields exit status 1), so this is not an error.
			p.proxyLogger.Debugf("<%s> process stopped by llama-swap: %v", p.id, waitErr)
		default:
			if exitErr, ok := waitErr.(*exec.ExitError); ok {
				p.proxyLogger.Debugf("<%s> process exited: code=%d, err=%v", p.id, exitErr.ExitCode(), waitErr)
			} else {
				p.proxyLogger.Debugf("<%s> process exited with error: %v", p.id, waitErr)
			}
		}
		close(cmdDone)
	}()

	abort := func(err error) startResult {
		p.killProcess(cmd, cmdCancel, cmdDone, 5*time.Second)
		return startResult{err: err}
	}
	prematureExit := func() startResult {
		cmdCancel()
		return startResult{err: fmt.Errorf("upstream command exited prematurely")}
	}

	if startCtx.Err() != nil {
		return abort(ErrStartAborted)
	}

	// A "none" checkEndpoint means the process is ready the moment it starts,
	// so skip the health-check polling entirely.
	if strings.TrimSpace(p.config.CheckEndpoint) == "none" {
		return startResult{cmd: cmd, cmdDone: cmdDone, cancel: cmdCancel, handlerFn: handlerFn}
	}

	// Wait 250ms for the command to start up before health checking
	select {
	case <-startCtx.Done():
		return abort(ErrStartAborted)
	case <-time.After(250 * time.Millisecond):
	}

	if err := p.pollHealth(startCtx, reverseProxy, healthCheckTimeout, cmdDone); err != nil {
		if errors.Is(err, errPrematureExit) {
			return prematureExit()
		}
		return abort(err)
	}

	return startResult{cmd: cmd, cmdDone: cmdDone, cancel: cmdCancel, handlerFn: handlerFn}
}

// buildHandler constructs the reverse proxy and its wrapping handler for the
// model's upstream. It is shared by doStart and doRestore so a restored process
// serves through the exact same proxy configuration as a fresh start. The
// returned handlerFn is ready to store on p.handler; reverseProxy is returned so
// callers that health-check through the proxy (pollHealth) can reuse it.
func (p *ProcessCommand) buildHandler() (http.HandlerFunc, *httputil.ReverseProxy) {
	proxyURL, err := url.Parse(p.config.Proxy)
	if err != nil {
		p.proxyLogger.Warnf("<%s> invalid proxy URL %q: %v", p.id, p.config.Proxy, err)
	}

	reverseProxy := httputil.NewSingleHostReverseProxy(proxyURL)
	reverseProxy.Transport = &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   time.Duration(p.config.Timeouts.Connect) * time.Second,
			KeepAlive: time.Duration(p.config.Timeouts.KeepAlive) * time.Second,
		}).DialContext,
		TLSHandshakeTimeout:   time.Duration(p.config.Timeouts.TLSHandshake) * time.Second,
		ResponseHeaderTimeout: time.Duration(p.config.Timeouts.ResponseHeader) * time.Second,
		ExpectContinueTimeout: time.Duration(p.config.Timeouts.ExpectContinue) * time.Second,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   10,
		IdleConnTimeout:       time.Duration(p.config.Timeouts.IdleConn) * time.Second,
		DisableKeepAlives:     p.config.DisableKeepAlives,
	}
	reverseProxy.ErrorHandler = newProxyErrorHandler(p.id, p.proxyLogger)
	reverseProxy.ModifyResponse = func(resp *http.Response) error {
		// Upstreams such as llama-server set their own CORS headers, and
		// ReverseProxy adds rather than replaces them, so both llama-swap's
		// and the upstream's would be sent. Keep only ours; see issue #85.
		swaputil.StripUpstreamCORSHeaders(resp.Header)

		if strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
			resp.Header.Set("X-Accel-Buffering", "no")
		}
		return nil
	}
	// httputil.ReverseProxy panics with http.ErrAbortHandler when the upstream
	// disconnects after response headers have been sent. Recover here so the
	// streaming termination is treated as a normal client/upstream disconnect.
	// see: https://github.com/golang/go/issues/23643
	handlerFn := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				if rec == http.ErrAbortHandler {
					p.proxyLogger.Infof("<%s> recovered from upstream disconnection during streaming", p.id)
				} else {
					p.proxyLogger.Warnf("<%s> recovered from panic: %v", p.id, rec)
				}
			}
		}()
		reverseProxy.ServeHTTP(w, r)
	})
	return handlerFn, reverseProxy
}

// pollHealth polls the upstream's check endpoint through reverseProxy until it
// returns 200, the context is cancelled, or the process exits. It shares the
// exact deadline + loading-progress semantics of a fresh start: the deadline is
// pushed back each time the reported progress value changes, so
// healthCheckTimeout bounds how long a load may go without progress rather than
// how long it may take in total. A message alone does not count: it can change
// while the load is stuck, e.g. a growing ETA.
//
// It returns nil when the upstream is ready; errPrematureExit when the process
// exits first; or an error (timeout / context cancel) otherwise.
func (p *ProcessCommand) pollHealth(ctx context.Context, reverseProxy *httputil.ReverseProxy, healthCheckTimeout time.Duration, cmdDone <-chan struct{}) error {
	deadline := time.Now().Add(healthCheckTimeout)
	var lastReport *LoadingProgress
	var lastProgress *float64
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-cmdDone:
			return errPrematureExit
		default:
		}

		if time.Now().After(deadline) {
			if lastProgress != nil {
				return fmt.Errorf("health check timed out after %v without loading progress", healthCheckTimeout)
			}
			return fmt.Errorf("health check timed out after %v", healthCheckTimeout)
		}

		// Tagged so the proxy ErrorHandler logs a not-yet-listening upstream
		// at debug rather than as a proxy error once per poll.
		checkCtx := context.WithValue(ctx, healthCheckKey{}, true)
		req, _ := http.NewRequestWithContext(checkCtx, "GET", p.config.CheckEndpoint, nil)
		rr := httptest.NewRecorder()
		reverseProxy.ServeHTTP(rr, req)
		resp := rr.Result()
		resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			p.proxyLogger.Infof("<%s> Health check passed on %s%s", p.id, p.config.Proxy, p.config.CheckEndpoint)
			return nil
		} else if ctx.Err() != nil {
			return ctx.Err()
		}

		if lp, ok := parseLoadingProgress(rr.Body.Bytes()); ok {
			if lp.Progress != nil && (lastProgress == nil || *lp.Progress != *lastProgress) {
				lastProgress = lp.Progress
				deadline = time.Now().Add(healthCheckTimeout)
			}
			if !lp.equal(lastReport) {
				lastReport = &lp
				p.setLoading(lp)
			}
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-cmdDone:
			return errPrematureExit
		case <-time.After(time.Second):
		}
	}
}

// doRestore brings a suspended process back to a ready state. Unlike doStart it
// does not spawn a new process: it re-acquires the accelerator state the sleeper
// offloaded, rebuilds the reverse-proxy handler, and health-checks through it.
// The existing cmd/cmdDone/cancel are echoed in the returned startResult so the
// run() loop latches them exactly as it does after a fresh start.
//
// On any failure (a sleeper error, a health-check timeout, or a premature exit)
// it returns a startResult with an err set and no cmd; the run() loop then falls
// back to starting a fresh process via doStart.
func (p *ProcessCommand) doRestore(startCtx context.Context, healthCheckTimeout time.Duration, cmd *exec.Cmd, cmdDone <-chan struct{}, cancel context.CancelFunc) startResult {
	if p.sleeper == nil {
		return startResult{err: fmt.Errorf("no sleeper configured for %s", p.id)}
	}

	p.proxyLogger.Infof("<%s> restoring suspended model", p.id)
	// Bound the resume to startCtx so a concurrent Stop (which cancels it) can
	// escape a stuck driver call instead of waiting out the full sleepTimeout.
	if err := p.sleeper.Resume(startCtx, cmd); err != nil {
		p.proxyLogger.Warnf("<%s> restore failed: %v; falling back to a fresh start", p.id, err)
		return startResult{err: fmt.Errorf("restore: %w", err)}
	}

	handlerFn, reverseProxy := p.buildHandler()
	if strings.TrimSpace(p.config.CheckEndpoint) == "none" {
		return startResult{cmd: cmd, cmdDone: cmdDone, cancel: cancel, handlerFn: handlerFn}
	}

	if err := p.pollHealth(startCtx, reverseProxy, healthCheckTimeout, cmdDone); err != nil {
		if errors.Is(err, errPrematureExit) {
			p.proxyLogger.Warnf("<%s> process exited during restore; falling back to a fresh start", p.id)
			return startResult{err: fmt.Errorf("upstream command exited prematurely")}
		}
		p.proxyLogger.Warnf("<%s> health check failed after restore: %v; falling back to a fresh start", p.id, err)
		return startResult{err: err}
	}

	p.proxyLogger.Infof("<%s> restored model is ready", p.id)
	return startResult{cmd: cmd, cmdDone: cmdDone, cancel: cancel, handlerFn: handlerFn}
}

// suspendProcess offloads the accelerator state of a running process via the
// configured sleeper. It returns nil on success, an error if the sleeper is
// unavailable or fails — in which case the caller should fall back to a normal
// kill.
func (p *ProcessCommand) suspendProcess(ctx context.Context, cmd *exec.Cmd) error {
	if p.sleeper == nil {
		return fmt.Errorf("no sleeper configured for %s", p.id)
	}
	p.proxyLogger.Infof("<%s> suspending model to offload accelerator state", p.id)
	if err := p.sleeper.Suspend(ctx, cmd); err != nil {
		p.proxyLogger.Warnf("<%s> suspend failed: %v; falling back to a kill", p.id, err)
		return err
	}
	p.proxyLogger.Infof("<%s> model suspended; accelerator state offloaded", p.id)
	return nil
}

// sendStopSignal runs the configured CmdStop (if any) or sends SIGTERM to
// the upstream process. Wired up as cmd.Cancel so it fires whenever the
// cmd's context is cancelled.
func (p *ProcessCommand) sendStopSignal(cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil {
		p.processLogger.Debugf("<%s> sendStopSignal() called with nil cmd or process, nothing to stop", p.id)
		return nil
	}
	pid := cmd.Process.Pid
	if p.config.CmdStop != "" {
		p.processLogger.Debugf("<%s> sendStopSignal() using CmdStop %q for pid %d", p.id, p.config.CmdStop, pid)
		stopArgs, err := config.SanitizeCommand(
			strings.ReplaceAll(p.config.CmdStop, "${PID}", fmt.Sprintf("%d", pid)),
		)
		if err == nil {
			p.processLogger.Debugf("<%s> sendStopSignal() running stop command: %s", p.id, strings.Join(stopArgs, " "))
			stopCmd := exec.Command(stopArgs[0], stopArgs[1:]...)
			stopCmd.Stderr = p.processLogger
			stopCmd.Stdout = p.processLogger
			// Bound the pipe copy so a CmdStop that backgrounds a child
			// holding stdout/stderr cannot block Run() indefinitely.
			stopCmd.WaitDelay = p.waitDelay
			stopCmd.Env = cmd.Env
			setProcAttributes(stopCmd)
			runErr := stopCmd.Run()
			// ErrWaitDelay is only returned when the stop command itself
			// succeeded, so it is not a failure to stop the process.
			if errors.Is(runErr, exec.ErrWaitDelay) {
				p.processLogger.Warnf("<%s> sendStopSignal() stop command exited but a child held its output open; output may be truncated", p.id)
				runErr = nil
			}
			if runErr != nil {
				p.processLogger.Errorf("<%s> sendStopSignal() stop command failed: %v", p.id, runErr)
			} else {
				p.processLogger.Debugf("<%s> sendStopSignal() stop command completed for pid %d", p.id, pid)
			}
			return runErr
		}
		// fall through to SIGTERM if sanitize failed
		p.processLogger.Errorf("<%s> sendStopSignal() failed to sanitize CmdStop %q: %v, falling back to terminateProcessTree", p.id, p.config.CmdStop, err)
	}
	// On Unix this SIGTERMs the whole process group so a forked grandchild
	// (e.g. a shell wrapper that backgrounds the real binary) is taken down
	// with the parent rather than orphaned.
	p.processLogger.Debugf("<%s> sendStopSignal() no CmdStop configured, calling terminateProcessTree for pid %d", p.id, pid)
	termErr := terminateProcessTree(cmd)
	if termErr != nil {
		p.processLogger.Errorf("<%s> sendStopSignal() terminateProcessTree failed for pid %d: %v", p.id, pid, termErr)
	}
	return termErr
}

// killProcess terminates the upstream process. The flow:
//
//  1. Send the graceful stop signal (CmdStop / SIGTERM) directly — NOT by
//     cancelling cmdCtx. Cancelling the context would start cmd.WaitDelay
//     immediately, which force-kills the process WaitDelay after the signal
//     and would silently cap gracefulTimeout at WaitDelay whenever
//     gracefulTimeout is the longer of the two.
//  2. We wait up to gracefulTimeout for the process to exit on its own.
//  3. If still alive, we SIGKILL the process group directly (Unix) so any
//     forked descendant is force-terminated alongside the parent.
//  4. We wait on cmdDone. cmd.WaitDelay (set when the cmd was built) is the
//     critical backstop here: once the process exits, if a forked grandchild
//     inherited the stdout/stderr pipes and is still holding them, the runtime
//     force-closes the pipes WaitDelay after the exit and cmd.Wait() unblocks.
//     Because we never cancelled the context, that WaitDelay timer measures
//     from process exit (see os/exec awaitGoroutines), not from this call.
//     Without WaitDelay this select would hang forever (the v219 bug).
//
// cancel() is still invoked (deferred) to release the context, but only after
// the process has exited and os/exec's ctx watcher has already torn down, so it
// never re-fires cmd.Cancel.
func (p *ProcessCommand) killProcess(cmd *exec.Cmd, cancel context.CancelFunc, cmdDone <-chan struct{}, gracefulTimeout time.Duration) {
	if cancel == nil {
		return
	}
	defer cancel()

	// Deliver CmdStop / SIGTERM in a goroutine so a slow or hanging CmdStop
	// cannot block the run() goroutine; the gracefulTimeout + Process.Kill
	// path below still guarantees teardown.
	if cmd != nil {
		go func() {
			p.proxyLogger.Debugf("[%s] sending stop signal with timeout %v", p.id, gracefulTimeout)
			if err := p.sendStopSignal(cmd); err != nil {
				p.proxyLogger.Warnf("[%s] stop signal failed: %v", p.id, err)
			}
		}()
	}

	timer := time.NewTimer(gracefulTimeout)
	defer timer.Stop()

	select {
	case <-cmdDone:
		return
	case <-timer.C:
	}

	if cmd != nil {
		// SIGKILL the whole process group on Unix so any descendant that
		// ignored or outlived the graceful signal is force-terminated too.
		_ = killProcessTree(cmd)
	}
	<-cmdDone
}

func (p *ProcessCommand) ID() string {
	return p.id
}

func (p *ProcessCommand) Run(timeout time.Duration) error {
	req := startReq{
		timeout: timeout,
		respond: make(chan error, 1),
		block:   true,
	}
	select {
	case p.startCh <- req:
	case <-p.parentCtx.Done():
		return fmt.Errorf("[%s] shutdown", p.id)
	}
	select {
	case err := <-req.respond:
		return err
	case <-p.parentCtx.Done():
		return fmt.Errorf("[%s] shutdown", p.id)
	}
}

// EnsureReady brings the process to a ready state and blocks until it is
// serving. See the Process interface for the full state-by-state contract.
//
// The send on startCh is the synchronisation point: it can only be received
// when the run loop is back at its select, so a stop that is still in progress
// naturally holds the request until the process is really stopped. Callers must
// not inspect State() first — that read races the run loop and is what caused
// issue #946.
func (p *ProcessCommand) EnsureReady(ctx context.Context, timeout time.Duration) error {
	req := startReq{
		timeout: timeout,
		respond: make(chan error, 1),
	}
	select {
	case p.startCh <- req:
	case <-ctx.Done():
		return ctx.Err()
	case <-p.parentCtx.Done():
		return fmt.Errorf("[%s] shutdown", p.id)
	}
	select {
	case err := <-req.respond:
		return err
	case <-ctx.Done():
		return ctx.Err()
	case <-p.parentCtx.Done():
		return fmt.Errorf("[%s] shutdown", p.id)
	}
}

func (p *ProcessCommand) WaitReady(ctx context.Context) error {
	req := waitReadyReq{respond: make(chan error, 1)}
	select {
	case p.waitReadyCh <- req:
	case <-ctx.Done():
		return ctx.Err()
	case <-p.parentCtx.Done():
		return fmt.Errorf("[%s] shutdown", p.id)
	}
	select {
	case err := <-req.respond:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (p *ProcessCommand) Stop(timeout time.Duration) error {
	return p.stop(timeout, false)
}

func (p *ProcessCommand) StopForce(timeout time.Duration) error {
	return p.stop(timeout, true)
}

func (p *ProcessCommand) stop(timeout time.Duration, force bool) error {
	req := stopReq{
		timeout: timeout,
		force:   force,
		respond: make(chan error, 1),
	}
	select {
	case p.stopCh <- req:
	case <-p.parentCtx.Done():
		return fmt.Errorf("[%s]: %w", p.id, ErrShutdown)
	}
	return <-req.respond
}

func (p *ProcessCommand) State() ProcessState {
	return p.Status().State
}

func (p *ProcessCommand) Status() Status {
	if st := p.status.Load(); st != nil {
		return *st
	}
	return Status{State: StateStopped}
}

func (p *ProcessCommand) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	fn := p.handler.Load()
	if fn == nil {
		swaputil.SendResponse(w, r, http.StatusServiceUnavailable, fmt.Sprintf("[%s] process is not ready", p.id))
		return
	}
	if p.config.Compat.IgnoreWebsockets && swaputil.IsWebSocketUpgrade(r) {
		(*fn)(w, r)
		return
	}
	p.inflight.Add(1)
	defer func() {
		p.lastUse.Store(time.Now().UnixNano())
		p.inflight.Add(-1)
	}()
	(*fn)(w, r)
}
