# NSAI

NSAI is a terminal app for building NationStates AI governance profiles and using a local OpenAI-compatible model, such as LM Studio, to advise on live NationStates issues.

It installs one primary command:

```powershell
nsai
```

Compatibility commands are also provided:

```powershell
ns-profile-builder
ns-live-advisor
```

## Nation Configs

NSAI can store one config per NationStates nation. The JSON config stores normal settings, while NationStates private API secrets and local-model API keys are stored separately in the OS credential store. On Windows, newly stored secrets default to the `windows-hello` backend: NSAI requires Windows Hello/user verification before it stores, reads, or deletes the secret. Use `--secret-backend keyring` only if you explicitly want the cross-platform keyring behavior instead.

Show config paths:

```powershell
poetry run nsai nation paths
```

Create or update a nation config:

```powershell
poetry run nsai nation set Oringrad `
  --user-agent 'InspyreSoftworksNationGM/0.1 contact:you@example.com nation:Oringrad' `
  --profile .\oringrad_governance_profile.json `
  --base-url 'http://localhost:1234/v1' `
  --model 'local-model-name' `
  --draft-dispatch `
  --draft-factbook `
  --auth-kind password `
  --password
```

When `--profile` is provided, NSAI copies that profile into its managed profile directory and saves the managed path in the nation config. Use `--move-profile` only when you want NSAI to move the original file instead of copying it.

`--password` prompts securely. For automation, `--password-stdin` reads the secret from stdin. The secret is not written to the JSON config. Use `--lm-api-key` or `--lm-api-key-stdin` to store a local-model API key behind the same secret backend.

Inspect a saved config without revealing secrets:

```powershell
poetry run nsai nation show Oringrad
poetry run nsai nation list
```

Remove a config and its stored secret:

```powershell
poetry run nsai nation remove Oringrad
```

## Install

From this folder:

```powershell
poetry install
```

Run the app:

```powershell
poetry run nsai --help
```

Build distributable artifacts:

```powershell
poetry build
```

## Profile Commands

Launch the Textual profile interview:

```powershell
poetry run nsai profile interview
```

Save only interview answers without AI-generated vision/constitution text:

```powershell
poetry run nsai profile interview --no-ai-append
```

Enrich an existing profile:

```powershell
poetry run nsai profile enrich .\oringrad_governance_profile.json
```

Preview a profile:

```powershell
poetry run nsai profile preview .\oringrad_governance_profile.json
```

The preview command renders the profile with Rich tables and panels instead of dumping raw JSON.

## Live Advisor Commands

Advisor-only mode is the default. Nothing is enacted unless you explicitly pass `--enact` or `--auto`.

```powershell
poetry run nsai advise --profile .\oringrad_governance_profile.json --show-issues
```

If a saved nation config exists, `--nation` can load its default profile, User-Agent, API version, and secure credential:

```powershell
poetry run nsai advise --nation Oringrad --show-issues
```

If you have saved a default nation with `--save-opts`, or only one nation config exists, bare `nsai advise` will use that saved nation. When NSAI automatically loads a saved config or profile, it prints the config/profile path before making live API calls.

Save the current advisor options as reusable defaults:

```powershell
poetry run nsai --save-opts advise --nation Oringrad `
  --profile .\oringrad_governance_profile.json `
  --base-url 'http://localhost:1234/v1' `
  --model 'local-model-name' `
  --draft-dispatch `
  --draft-factbook `
  --show-issues
```

`--save-opts` stores the default nation in the program config and stores advisor defaults in that nation's config: profile path, strategy, model base URL, model name, display toggles, local-AI toggle, audit log path, and publication draft preferences. It copies a CLI-provided profile into managed storage. If `--lm-api-key` is provided with `--save-opts`, the key is stored in the OS credential store using `--secret-backend` or the platform default. It does not store `--enact`, `--auto`, `--override-red-line`, or `--no-nation-config`.

Skip saved nation config and use only CLI/env values:

```powershell
poetry run nsai advise --nation Oringrad --no-nation-config
```

Use deterministic fallback logic without local AI:

```powershell
poetry run nsai advise --nation Oringrad --no-ai
```

By default, the advisor stores issue choices and per-issue advice in a SQLite cache under the NSAI config directory. If the same live issue IDs are present on a later run, NSAI reuses the saved "most important issue" choice. If the chosen issue already has saved advice, NSAI reuses that recommendation instead of contacting the local AI again. If a new issue ID appears, the issue choice is recalculated, but saved advice for any selected issue ID is still reused.

Force a fresh model pass for the current issue set:

```powershell
poetry run nsai advise --nation Oringrad --refresh-advice
```

When an issue action is enacted or dismissed through NSAI, the advice cache records the raw NationStates response plus extracted effect/stat and headline fields when they are present. AI token usage for issue selection and advice generation is also recorded when the OpenAI-compatible server reports it.

Backfill publication drafts from older enacted audit records:

```powershell
poetry run nsai publications backfill --audit-log ns_governor_audit.jsonl
poetry run nsai publications backfill --audit-log ns_governor_audit.jsonl --execute
```

The preview form lists enacted recommendations with dispatch/factbook drafts that have not been posted yet. `--execute` creates missing pages and appends publication backfill records to the audit log so successful posts are not duplicated. NationStates may limit how many announcement pages a nation can create in a short period, so NSAI shows a progress bar and waits between posts for the same nation. The default delay is 300 seconds; override it with `--cooldown-seconds SECONDS`. If NationStates still returns a publication cooldown error, NSAI waits and retries according to `--cooldown-retries`.

Ask the AI to also produce publication-ready dispatch/factbook text:

```powershell
poetry run nsai advise --nation Oringrad --draft-dispatch --draft-factbook
```

`--draft-dispatch` requires a dispatch title and body in the AI recommendation. `--draft-factbook` asks the AI to decide whether the issue action is pertinent to durable national lore, institutions, laws, or statistics; if so, it drafts a factbook entry, otherwise it explains why no factbook update is needed. These can also be saved as per-nation defaults with `nsai nation set`. When the recommendation is actually applied with `--enact` or allowed `--auto`, NSAI posts the first requested publication page through the NationStates dispatch private command and leaves any additional page pending for `nsai publications backfill`. Advisor-only runs print the drafts but do not publish pages.

Manually apply a validated recommendation. This can enact a selected issue option or dismiss the issue when the AI recommends dismissal:

```powershell
poetry run nsai advise --profile .\oringrad_governance_profile.json --enact
```

Allow profile-controlled auto-enactment:

```powershell
poetry run nsai advise --profile .\oringrad_governance_profile.json --auto
```

`--auto` is not an override switch. It only allows automatic action when a loaded profile has `enactment_mode` set to `auto_enact_high_confidence`, `auto_enact_unless_red_line`, or `fully_autonomous`; the recommendation is not a fallback; no red line is triggered; and the confidence meets the profile's `minimum_confidence_to_enact`. With `auto_enact_high_confidence`, confidence must also be at least `0.85`. If cached advice is stale or too conservative, use `--refresh-advice` to ask the model again.

`--flag-display ascii` renders the nation's flag image as ASCII art after NSAI loads the NationStates nation. Use `--flag-display banner` for a simple text banner instead.

## Environment Variables

NationStates requires an informative User-Agent. In PowerShell:

```powershell
$env:NS_USER_AGENT='InspyreSoftworksNationGM/0.1 contact:you@example.com nation:Oringrad'
```

Private issue access needs one of these:

```powershell
$env:NS_PASSWORD='your-password'
$env:NS_AUTOLOGIN='your-autologin-token'
$env:NS_PIN='your-pin'
```

Environment variables still work and override saved nation config values for the current process.

Optional settings:

```powershell
$env:NS_NATION='Oringrad'
$env:NS_API_VERSION='12'
$env:LM_STUDIO_BASE_URL='http://localhost:1234/v1'
$env:LM_STUDIO_MODEL='local-model-name'
$env:LM_STUDIO_API_KEY='lm-studio'
```

`.env.example` documents the variables, but NSAI does not load `.env` files by itself. Environment variables override saved config values for that process.

## LM Studio

The profile enricher and live advisor use the OpenAI Python client against an OpenAI-compatible local endpoint. By default, NSAI expects LM Studio at:

```text
http://localhost:1234/v1
```

If `LM_STUDIO_MODEL` is not set, NSAI asks the local server for the first available model.

You can also configure these through `nsai nation set --base-url --model` or save the current advisor values with `nsai advise --base-url --model --save-opts`.

## Safety

NSAI instructs the AI to summarize the selected issue and every available option before recommending an action. The AI can recommend either enacting a real option or dismissing the issue. Dismissal uses the NationStates API dismissal option, `option=-1`.

NSAI validates that AI recommendations use real live issue IDs and either a real option ID or the dismissal option before any action path can continue. Fallback recommendations are for review only and are not allowed to enact or dismiss issues.

The default live advisor mode is audit/advice only. Use `--enact` for manual application or `--auto` for profile-controlled autonomy.

Dispatch and factbook pages are created through the NationStates `dispatch` private command using `dispatch=add`, `title`, `text`, `category`, `subcategory`, and the documented prepare/execute flow. NSAI only posts publication pages after a successful issue action, so generated text from advisor-only runs stays review-only.
