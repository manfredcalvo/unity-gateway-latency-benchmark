# Methodology

## Question being answered

> When an application calls a Bedrock model through Databricks Unity Gateway instead of
> calling Bedrock directly, how much latency is added, and how much does each gateway
> feature add on top?

The overhead is in milliseconds. The model's own latency is hundreds of milliseconds
and varies by tens to hundreds of ms between identical requests. The whole method exists to
measure a small signal reliably under that much noise.

## Controls

| Risk | Control |
|---|---|
| Provider drift over time (Bedrock load, time of day) biases whichever target ran later | **Randomized complete blocks:** every round sends one request to every target, in a freshly shuffled order |
| Cold connections (DNS, TCP, TLS) inflate early samples | Persistent keep-alive clients per target, plus **warmup rounds** that are recorded and then discarded |
| Auth token refresh shows up as "gateway latency" | Databricks credentials are resolved before the timer starts. SigV4 signing for Bedrock is inside the timer: it's a real client cost, well under 1 ms. |
| Hidden client retries | boto3 `max_attempts=1`, and no retries in httpx |
| Different work per path | Same model, prompt, `max_tokens`, temperature 0 on every path |
| Network distance confounds gateway cost | Client, workspace, and Bedrock in the **same AWS region**. The server-side breakdown separates the two. |
| Rate limits accidentally throttle | Limits set at 100k QPM: evaluated on every request, never triggered |
| Inference-table / usage writes | Both are asynchronous in the gateway. The benchmark measures their effect on the request path, which is the only part a caller sees. |

## Metrics

* `latency_ms`: wall clock from send to full response read (`perf_counter_ns`).
* `ttfb_ms`: response headers received.
* `ttft_ms` (`--stream`): first content token received. This is the metric that matters for chat UX.
* `provider_latency_ms`: Bedrock's own `metrics.latencyMs` (direct path only).

## Estimators

1. **Percentile deltas** `Pq(target) − Pq(direct)` for q ∈ {50, 90, 99}, each with a 95% bootstrap
   CI (independent resampling, 2,000 replicates). This is the form the product team's numbers take.
2. **Paired median delta** `median_r(target_r − direct_r)`. Requests in a round run within
   seconds of each other, so slow provider drift cancels. This is usually the tightest estimate of
   typical added latency.
3. **Server-side gateway-internal time** from `system.ai_gateway.usage`:
   `latency_ms − Σ routing_information.attempts[].latency_ms`. This is the time spent inside the
   gateway, excluding Bedrock's time and the client network. Matched to client rows through the
   `Databricks-Ai-Gateway-Request-Tags` header (`ugbench_run`, `ugbench_target`, `ugbench_round`).

## Sample size

* Client-side **P50** overhead of a few ms needs roughly 1,000 or more rounds to resolve. The paired median
  converges faster.
* Client-side **P99** deltas are dominated by the model's tail (a few slow generations). With 1,000
  rounds, the P99 rests on about 10 samples per target, and the CI is tens of ms wide. Use ≥ 2,000 rounds.
  For tail overhead, rely on the **server-side** gateway-internal P99, which excludes model time.
* The report flags P99 values based on fewer than about 20 tail samples.

## Interpreting results honestly

* A CI that contains 0 means the overhead is **smaller than the measurement can resolve**. That is a
  valid result: report it as "not distinguishable from zero at n = …".
* Negative point estimates happen with noise. Don't report them as "the gateway is faster".
* Client-side and server-side numbers answer different questions. Present both:
  "your application will see +X ms end to end; of that, the gateway itself accounts for Y ms."
* Product team reference (gateway-internal, internal testing at ~100 QPS):

  | Configuration | P50 | P99 |
  |---|---|---|
  | No gateway features | 4 ms | 9 ms |
  | Usage tracking | 7 ms | 15 ms |
  | Inference tables | 7 ms | 15 ms |
  | Rate limits | 6 ms | 9 ms |
