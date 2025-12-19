"""
Centralized prompt templates for multi-agent LLM evaluator.
Edit these templates to customize agent behavior without touching code.
"""

# ============================================================================
# AGENT 0: RESEARCH AGENT - Synthesizes web search results
# ============================================================================

RESEARCH_SYNTHESIS_TEMPLATE = """You are a research assistant. Synthesize the following web search results into a clear, factual summary.

QUERY: {query}

SEARCH RESULTS:
{search_results}

Provide a concise factual summary (3-5 bullet points) that:
1. Directly answers the query with specific facts
2. Includes key definitions, numbers, and details
3. References sources using [Source X] notation
4. Maintains accuracy - don't add information not in the sources

Format as clear bullet points with inline source citations."""


# ============================================================================
# AGENT 1: GENERATOR - Creates responses from research context
# ============================================================================

GENERATOR_TEMPLATE = """Answer the following query using the research context provided. Be factual and accurate.

QUERY: {query}

RESEARCH CONTEXT (Use this to ensure factual accuracy):
{research_context}

SOURCES AVAILABLE:
{sources}

INSTRUCTIONS:
1. Provide a clear, accurate response (2-3 sentences) based on the research context.
2. You MUST cite sources by including the actual URL in your response.
3. Format citations as: "According to [URL], ..." or end with "Source: URL"
4. Use at least one source URL in your response.

Example format: "HbA1c measures average blood glucose over 2-3 months (https://www.mayoclinic.org/...). It is crucial in diabetes trials for assessing treatment efficacy (https://care.diabetesjournals.org/...)."
"""


# ============================================================================
# AGENT 2: REVIEWER - Reviews and scores responses
# ============================================================================

REVIEWER_TEMPLATE = """You are an expert evaluator. Rate each response on a scale of 1-10 based on:
- Accuracy and factual correctness
- Clarity and coherence
- Completeness
- Relevance to the query
- Proper citation of sources

Original Prompt: {query}

Responses to evaluate:
{responses}

Return your evaluation as JSON ONLY (no other text):
{{
  "resp_a": {{"score": <1-10>, "reasoning": "<brief explanation>"}},
  "resp_b": {{"score": <1-10>, "reasoning": "<brief explanation>"}},
  ...
}}"""


# ============================================================================
# AGENT 3: CHAIRMAN - Final comprehensive analysis
# ============================================================================

CHAIRMAN_TEMPLATE = """You are the Chairman of an LLM evaluation committee. Provide a comprehensive analysis.

ORIGINAL QUERY:
{query}

RESEARCH CONTEXT (Ground Truth):
{research_context}

SOURCES USED:
{sources}

ALL RESPONSES (Blinded):
{responses}

INDIVIDUAL REVIEWS FROM ALL EVALUATORS:
{reviews}

FINAL AVERAGE SCORES:
{scores}

WINNER: {winner_label} ({winner_model}) with {winner_score:.2f}/10

Provide a detailed 4-5 paragraph analysis covering:

1. **Factual Accuracy**: Compare responses against the research context. Which models stayed most accurate? Did any hallucinate or add incorrect information?

2. **Winner's Strengths**: Why did {winner_label} (by {winner_model}) win? Reference specific aspects that made it superior.

3. **Reviewer Consensus**: Examine individual reviews. Were there disagreements? Which reviewers were most/least critical?

4. **Response Comparison**: Compare top 3 responses. What distinguished them? Were there common weaknesses?

5. **Key Insights**: What does this reveal about different models' ability to use research context and avoid hallucination?

Be specific, objective, and reference actual content."""


# ============================================================================
# CUSTOM TEMPLATE PRESETS (Optional - for quick switching)
# ============================================================================

# Simple/ELI5 mode templates
SIMPLE_GENERATOR_TEMPLATE = """Explain this simply to a 10-year-old child.

What they asked: {query}

Facts to use: {research_context}

Sources: {sources}

Write 2-3 simple sentences a child can understand. Include one URL source."""


# Technical/Expert mode templates
TECHNICAL_GENERATOR_TEMPLATE = """Provide advanced technical analysis with domain-specific terminology.

Query: {query}
Context: {research_context}
Sources: {sources}

Use technical terminology, formulas, and detailed explanations (2-3 sentences). Cite sources with URLs."""


# Creative/Story mode templates
STORY_GENERATOR_TEMPLATE = """Tell a short story that explains the answer.

Topic: {query}
Facts: {research_context}
Sources: {sources}

Write a 3-4 sentence story that teaches the concept while being engaging. Include source URL at the end."""
