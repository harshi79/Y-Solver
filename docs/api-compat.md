# Wire-protocol compatibility notes

Y-Solver implements the *de facto* HTTP contract shared by the mainstream captcha-solving
services, so an existing client usually needs only a base-URL change. This page lists what
is implemented, what is deliberately different, and the small deviations you may trip over.

## Implemented surface

### `POST /in.php` (also `GET`)

| Parameter | Notes |
|---|---|
| `key` / `clientKey` / `apiKey` | Any of the aliases is accepted. |
| `method` | `post`, `base64`, `file`, `image`, `imagetotext` → image captcha. Interactive names → refused. Omitted → treated as an image upload. |
| `file` (multipart) | Also accepted under the names `image`, `captcha`. Multiple files: the first is used. |
| `body` / `image` / `captcha` / `base64` | Base64 payload. Raw base64, URL-safe base64, unpadded base64, and `data:image/png;base64,…` URLs all work. |
| raw request body | If the body is a recognisable image signature, it is used directly, whatever the content type says. |
| `numeric` / `numerals` / `digits` | Restricts the alphabet to `0-9`. |
| `min_len` / `max_len` | Applied to the returned text: padded by repeating the last character, or trimmed. Best-effort — an OCR engine cannot invent a character it could not read. |
| `charset` / `characters` / `alphabet` / `whitelist` | Custom alphabet. |
| `delay` | ≥ 100 s schedules the job (a timer, not a paid tier). |
| `comment`, `textinstructions` | Accepted and ignored, for client compatibility. |
| `pingback` / `proxy` / … | Accepted and ignored. Y-Solver never makes outbound requests, so no proxy or callback is possible. |

Responses are plain text: `OK|<id>` or an `ERROR_*` token, exactly as clients expect.

### `GET /res.php` (also `POST`)

| Parameter | Response |
|---|---|
| `action=get`, `id=<id>` | `OK|<text>` · `CAPCHA_NOT_READY` · `ERROR_*` |
| `action=getbalance` | `OK|$9999.00` (cosmetic — configurable via `YSOLVER_BALANCE`) |
| `action=reportbad` / `report` | `OK|OK` (accepted; there is no human queue to flag to) |
| unknown action | `ERROR_BAD_ACTION` |

The historical misspelling `CAPCHA_NOT_READY` is returned on purpose — clients match on it.

### `POST /createTask` and `POST /getTaskResult`

* `task` may be a nested object or a JSON string, or the fields may be flat in the root.
* `task.type` naming conventions are matched loosely: `ImageToTextTask`,
  `ImageToTextTaskProxyless`, `ImageToTextTaskMugshot` … all route to the image engine.
* Field aliases: `body` / `image` / `captcha` / `base64`.
* Result shape: `{"errorId":0,"status":"processing"}` →
  `{"errorId":0,"status":"ready","solution":{"text":…,"confidence":…,"backend":…,"solveMs":…}}`.
  `confidence`, `backend` and `solveMs` are additions — extra keys, safe to ignore.
* Failures: `{"errorId":1,"errorCode":"ERROR_…","errorDescription":"…"}`.

## Deliberate differences

| Area | Paid services | Y-Solver |
|---|---|---|
| Interactive challenges | Solve them (humans/farms) | `ERROR_METHOD_NOT_SUPPORTED` with an explanation |
| Pricing/quota/billing | Per 1 000, minimum top-ups | None. `getbalance` is cosmetic. |
| Pingback/callbacks | Outbound HTTP callbacks | Not implemented — the server never makes outbound requests |
| Proxies | Passed to the solving infrastructure | Irrelevant; there is no remote infrastructure |
| `reportbad` | Enters a human review flow | Accepted, no-op |
| Result text casing | Upper-case usually | Whatever the image shows; engines preserve case |
| Job ids | Provider-specific (often long integers) | 9-digit numeric ids, so strict client parsers keep working |

## Client check-list when swapping in Y-Solver

1. Replace the API base URL only.
2. Point `key`/`clientKey` at one of `YSOLVER_API_KEYS` (`demo` by default). Set
   `YSOLVER_REQUIRE_KEY=0` if the client cannot send a key at all.
3. If the client sends `numeric=1`, keep it — it improves accuracy.
4. Polling intervals are fine as-is; typical solves land in 20–40 ms, so even a 5 s first
   poll resolves on the first try.
5. If a client branches on specific error codes, note that unreadable images produce
   `ERROR_CAPTCHA_UNSOLVABLE`, and interactive requests produce
   `ERROR_METHOD_NOT_SUPPORTED`.
