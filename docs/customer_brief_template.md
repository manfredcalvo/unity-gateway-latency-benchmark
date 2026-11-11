# Databricks Unity Gateway: latency impact on Amazon Bedrock calls

**Prepared for:** <Customer> · **Date:** <date> · **Run ID:** `<run_id>` (raw data available on request)

## Bottom line

Routing <model> calls through Databricks Unity Gateway added a median of **<X> ms** per request
(95% CI <lo> to <hi> ms). That is **<Y>%** of the <P50> ms end-to-end latency of calling Bedrock
directly. Enabling governance features (inference tables, rate limits, usage tracking) added
**<Z> ms** more at the median.

## What we tested

* Same model (`<model_id>`), same prompt, same settings, region `<region>`.
* Client in `<where>`, in the same AWS region as the Databricks workspace and Bedrock.
* <N> rounds. Each round called every configuration once, in random order, so all configurations
  saw the same Bedrock conditions.

## Results

| Path | P50 | P90 | P99 | Added vs. direct (median, 95% CI) |
|---|---|---|---|---|
| Bedrock direct | | | | n/a |
| Unity Gateway, no optional features | | | | |
| + inference table | | | | |
| + rate limits | | | | |
| + all features | | | | |

<insert charts/overhead_vs_direct.png>

**Inside the gateway** (server-side, from `system.ai_gateway.usage`): P50 <a> ms, P99 <b> ms.
The rest of the client-observed difference is network transit between the client, Databricks,
and Bedrock.

For reference, Databricks' internal measurement of gateway overhead: P50 4 ms / P99 9 ms with no
features, and 6 to 7 ms P50 / 9 to 15 ms P99 with usage tracking, inference tables, or rate limits enabled.

## What you get for those milliseconds

Central access control in Unity Catalog, usage and cost tracking per user/team, full
request/response audit logging to Delta tables, rate limits, fallbacks, and guardrails, applied to
every model provider from one place.

## Reproduce it yourself

The benchmark is a small open Python tool (`ugbench`). Point it at your workspace and AWS
account and run `ugbench setup && ugbench run`. All raw per-request data is kept, so every number above
can be recomputed.
