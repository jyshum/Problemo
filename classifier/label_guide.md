# Label Guide for Pain Point Classifier

Review this guide before labeling any posts. Edit it to add examples
you find during labeling.

---

## Classes

### WORKFLOW_PAIN
Person describes a repetitive task done manually or inefficiently,
with no satisfying solution currently in place. Also includes frustration
with a specific named product where a feature is missing, broken, or
worse than expected.

**Signal phrases:** "every week I", "I manually", "takes hours",
"still using spreadsheets", "no tool for", "wish there was a way to",
"have to", "tedious", "time-consuming", "[product] is broken",
"[product] is missing", "wish it had", "doesn't work", "terrible"

**Positive examples:**
- "Every Monday I spend 3 hours copying data from emails into a spreadsheet. There has to be a better way."
- "I manually track all my client invoices in Excel. It takes forever and I always make mistakes."
- "Our team still uses paper forms for patient intake. No digital solution exists that works for our workflow."
- "Notion's table view has been broken for weeks and they haven't fixed it."
- "Slack's search is terrible. I can never find old messages."

**Negative examples (don't label as WORKFLOW_PAIN):**
- Asking if a tool exists → TOOL_REQUEST
- Generic venting with no workflow or product frustration described → NOISE

---

### TOOL_REQUEST
Person explicitly asks whether a tool exists for a task, or directly
requests that one be built.

**Signal phrases:** "is there an app that", "is there a tool for",
"does a tool exist for", "someone should build", "does anyone know of",
"looking for software that", "any recommendations for"

**Positive examples:**
- "Is there an app that automatically categorizes my bank transactions by project?"
- "Does a tool exist for tracking which clients have seen which proposals?"
- "Someone should build a plugin that syncs Notion with Google Calendar automatically."

**Negative examples:**
- Already describes a manual process without asking → WORKFLOW_PAIN

---

### NOISE
Post does not describe an actionable pain point. Includes: general
venting, career advice, off-topic discussion, promotional content,
news sharing, philosophical questions.

**Examples:**
- "Anyone else just burned out from tech work?"
- "What's the best programming language to learn in 2024?"
- "I just launched my SaaS, check it out! [link]"
- "Remote work is destroying work-life balance."

---

## Quick Reference

| If the post... | Label |
|---|---|
| Describes a manual/inefficient process, no tool mentioned | WORKFLOW_PAIN |
| Frustrated with a specific named product (feature missing/broken) | WORKFLOW_PAIN |
| Asks whether a tool exists or requests one | TOOL_REQUEST |
| Everything else | NOISE |

## Edge Cases

- Post does BOTH: describes a workflow pain AND asks for a tool → **TOOL_REQUEST** (more specific)
- Post mentions a product name but is venting generally → **NOISE** (no specific feature complaint)
- Post is very short (< 2 sentences) with no clear pain → **NOISE**
